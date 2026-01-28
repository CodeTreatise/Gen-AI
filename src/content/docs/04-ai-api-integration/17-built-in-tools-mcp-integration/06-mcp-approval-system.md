---
title: "MCP Approval System"
---

# MCP Approval System

## Introduction

The MCP approval system provides control over which tool calls are executed automatically versus requiring human approval. This is critical for security-sensitive operations and maintaining oversight of AI actions.

### What We'll Cover

- Understanding require_approval settings
- Handling mcp_approval_request
- Submitting mcp_approval_response
- Configuring trusted servers
- Building approval workflows

### Prerequisites

- MCP fundamentals
- MCP workflow understanding
- Security considerations

---

## Understanding require_approval Settings

### Approval Modes

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

class ApprovalMode(Enum):
    NEVER = "never"      # Auto-approve all calls
    ALWAYS = "always"    # Require approval for every call
    AUTO = "auto"        # Let model decide based on risk


# Server configurations with different approval modes
APPROVAL_EXAMPLES = {
    "weather": {
        "type": "mcp",
        "server_label": "weather",
        "server_url": "https://mcp.example.com/weather",
        "require_approval": "never"  # Safe, read-only data
    },
    "file_system": {
        "type": "mcp",
        "server_label": "file_system",
        "server_url": "https://mcp.example.com/files",
        "require_approval": "always"  # Write operations need approval
    },
    "database": {
        "type": "mcp",
        "server_label": "database",
        "server_url": "https://mcp.example.com/database",
        "require_approval": "auto"  # Model decides based on operation
    }
}
```

### Approval Configuration

```python
@dataclass
class ApprovalConfig:
    """Configuration for MCP approval behavior."""
    
    mode: ApprovalMode = ApprovalMode.AUTO
    auto_approve_tools: List[str] = field(default_factory=list)
    always_require_tools: List[str] = field(default_factory=list)
    timeout_seconds: int = 300  # 5 minute approval timeout
    default_on_timeout: str = "deny"  # deny or approve
    
    def should_auto_approve(self, tool_name: str) -> bool:
        """Check if tool should be auto-approved."""
        
        if self.mode == ApprovalMode.NEVER:
            return True
        
        if self.mode == ApprovalMode.ALWAYS:
            return tool_name in self.auto_approve_tools
        
        # AUTO mode
        if tool_name in self.always_require_tools:
            return False
        
        if tool_name in self.auto_approve_tools:
            return True
        
        return False  # Default to requiring approval in auto mode
    
    def to_tool_config(self, base_config: dict) -> dict:
        """Add approval settings to tool config."""
        
        config = base_config.copy()
        config["require_approval"] = self.mode.value
        
        return config


# Usage
approval_config = ApprovalConfig(
    mode=ApprovalMode.AUTO,
    auto_approve_tools=["get_weather", "list_files", "read_file"],
    always_require_tools=["delete_file", "execute_query", "send_email"],
    timeout_seconds=120
)

print(approval_config.should_auto_approve("get_weather"))  # True
print(approval_config.should_auto_approve("delete_file"))  # False
```

---

## Handling mcp_approval_request

### Basic Approval Request Handling

```python
from openai import OpenAI
from datetime import datetime

client = OpenAI()

# Request that triggers approval
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "mcp",
        "server_label": "database",
        "server_url": "https://mcp.example.com/database",
        "require_approval": "always"
    }],
    input="Delete all inactive users from the database"
)

# Check for approval requests
for item in response.output:
    if hasattr(item, 'type') and item.type == 'mcp_approval_request':
        print(f"Approval requested for: {item.name}")
        print(f"Arguments: {item.arguments}")
        print(f"Approval ID: {item.id}")
```

### Approval Request Manager

```python
@dataclass
class ApprovalRequest:
    """Pending approval request."""
    
    id: str
    tool_name: str
    server_label: str
    arguments: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    status: str = "pending"  # pending, approved, denied, expired
    reason: Optional[str] = None


class ApprovalRequestHandler:
    """Handle incoming approval requests."""
    
    def __init__(self, timeout_seconds: int = 300):
        self.timeout = timeout_seconds
        self.requests: Dict[str, ApprovalRequest] = {}
    
    def process_response(self, response) -> List[ApprovalRequest]:
        """Extract approval requests from response."""
        
        requests = []
        now = datetime.now()
        
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'mcp_approval_request':
                request = ApprovalRequest(
                    id=item.id,
                    tool_name=item.name,
                    server_label=getattr(item, 'server_label', ''),
                    arguments=getattr(item, 'arguments', {}),
                    created_at=now,
                    expires_at=now + timedelta(seconds=self.timeout)
                )
                
                self.requests[request.id] = request
                requests.append(request)
        
        return requests
    
    def get_pending(self) -> List[ApprovalRequest]:
        """Get all pending requests."""
        
        self._expire_old_requests()
        
        return [
            r for r in self.requests.values()
            if r.status == "pending"
        ]
    
    def approve(self, request_id: str, reason: str = None) -> bool:
        """Approve a request."""
        
        if request_id not in self.requests:
            return False
        
        request = self.requests[request_id]
        
        if request.status != "pending":
            return False
        
        if datetime.now() > request.expires_at:
            request.status = "expired"
            return False
        
        request.status = "approved"
        request.reason = reason
        return True
    
    def deny(self, request_id: str, reason: str = None) -> bool:
        """Deny a request."""
        
        if request_id not in self.requests:
            return False
        
        request = self.requests[request_id]
        
        if request.status != "pending":
            return False
        
        request.status = "denied"
        request.reason = reason
        return True
    
    def _expire_old_requests(self):
        """Expire old pending requests."""
        
        now = datetime.now()
        for request in self.requests.values():
            if request.status == "pending" and now > request.expires_at:
                request.status = "expired"
    
    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a specific request."""
        return self.requests.get(request_id)


# Usage
handler = ApprovalRequestHandler(timeout_seconds=300)

# Process response for approval requests
# requests = handler.process_response(response)
# 
# for req in requests:
#     print(f"Tool: {req.tool_name}")
#     print(f"Args: {req.arguments}")
#     
#     # Approve or deny
#     handler.approve(req.id, reason="User confirmed action")
```

---

## Submitting mcp_approval_response

### Basic Approval Response

```python
# After getting approval request, submit response
def submit_approval(
    client: OpenAI,
    response_id: str,
    approval_id: str,
    approved: bool,
    reason: str = None
) -> dict:
    """Submit approval response to continue execution."""
    
    approval_response = {
        "type": "mcp_approval_response",
        "approval_id": approval_id,
        "approved": approved
    }
    
    if reason:
        approval_response["reason"] = reason
    
    # Continue the response with approval
    continued = client.responses.create(
        model="gpt-4o",
        previous_response_id=response_id,
        input=[approval_response]
    )
    
    return {
        "output": continued.output_text,
        "approved": approved
    }


# Usage
# result = submit_approval(
#     client,
#     response_id=response.id,
#     approval_id=approval_request.id,
#     approved=True,
#     reason="Administrator approved database operation"
# )
```

### Approval Response Builder

```python
@dataclass
class ApprovalResponse:
    """Response to an approval request."""
    
    approval_id: str
    approved: bool
    reason: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to API format."""
        response = {
            "type": "mcp_approval_response",
            "approval_id": self.approval_id,
            "approved": self.approved
        }
        
        if self.reason:
            response["reason"] = self.reason
        
        return response


class ApprovalResponseBuilder:
    """Build and submit approval responses."""
    
    def __init__(self):
        self.client = OpenAI()
        self.responses: List[ApprovalResponse] = []
    
    def approve(
        self,
        request: ApprovalRequest,
        approved_by: str = None,
        reason: str = None
    ) -> ApprovalResponse:
        """Create approval response."""
        
        response = ApprovalResponse(
            approval_id=request.id,
            approved=True,
            reason=reason,
            approved_by=approved_by,
            approved_at=datetime.now()
        )
        
        self.responses.append(response)
        return response
    
    def deny(
        self,
        request: ApprovalRequest,
        denied_by: str = None,
        reason: str = None
    ) -> ApprovalResponse:
        """Create denial response."""
        
        response = ApprovalResponse(
            approval_id=request.id,
            approved=False,
            reason=reason or "Request denied",
            approved_by=denied_by,
            approved_at=datetime.now()
        )
        
        self.responses.append(response)
        return response
    
    def submit(
        self,
        response_id: str,
        approval_response: ApprovalResponse
    ) -> dict:
        """Submit approval response to API."""
        
        continued = self.client.responses.create(
            model="gpt-4o",
            previous_response_id=response_id,
            input=[approval_response.to_dict()]
        )
        
        return {
            "output": continued.output_text,
            "approval_id": approval_response.approval_id,
            "approved": approval_response.approved
        }
    
    def submit_batch(
        self,
        response_id: str,
        approval_responses: List[ApprovalResponse]
    ) -> dict:
        """Submit multiple approval responses."""
        
        inputs = [r.to_dict() for r in approval_responses]
        
        continued = self.client.responses.create(
            model="gpt-4o",
            previous_response_id=response_id,
            input=inputs
        )
        
        return {
            "output": continued.output_text,
            "approvals_submitted": len(approval_responses)
        }


# Usage
builder = ApprovalResponseBuilder()

# Approve a request
# approval = builder.approve(
#     request=pending_request,
#     approved_by="admin@example.com",
#     reason="Verified safe operation"
# )
# 
# result = builder.submit(response.id, approval)
```

---

## Configuring Trusted Servers

### Trusted Server List

```python
@dataclass
class TrustedServer:
    """A trusted MCP server configuration."""
    
    label: str
    url: str
    trusted_tools: List[str] = field(default_factory=list)
    trust_level: str = "full"  # full, partial, read_only
    added_at: datetime = field(default_factory=datetime.now)
    added_by: Optional[str] = None


class TrustManager:
    """Manage trusted MCP servers."""
    
    def __init__(self):
        self.trusted_servers: Dict[str, TrustedServer] = {}
        self.trust_history: List[dict] = []
    
    def add_trusted_server(
        self,
        label: str,
        url: str,
        trusted_tools: List[str] = None,
        trust_level: str = "full",
        added_by: str = None
    ):
        """Add a trusted server."""
        
        server = TrustedServer(
            label=label,
            url=url,
            trusted_tools=trusted_tools or [],
            trust_level=trust_level,
            added_by=added_by
        )
        
        self.trusted_servers[label] = server
        
        self.trust_history.append({
            "action": "add",
            "server": label,
            "by": added_by,
            "timestamp": datetime.now().isoformat()
        })
    
    def remove_trusted_server(self, label: str, removed_by: str = None):
        """Remove a trusted server."""
        
        if label in self.trusted_servers:
            del self.trusted_servers[label]
            
            self.trust_history.append({
                "action": "remove",
                "server": label,
                "by": removed_by,
                "timestamp": datetime.now().isoformat()
            })
    
    def is_trusted(
        self,
        server_label: str,
        tool_name: str = None
    ) -> bool:
        """Check if server/tool is trusted."""
        
        if server_label not in self.trusted_servers:
            return False
        
        server = self.trusted_servers[server_label]
        
        # Full trust
        if server.trust_level == "full":
            return True
        
        # Partial trust - check specific tools
        if tool_name and server.trusted_tools:
            return tool_name in server.trusted_tools
        
        # Read-only trust - no write operations
        if server.trust_level == "read_only":
            return self._is_read_operation(tool_name)
        
        return False
    
    def _is_read_operation(self, tool_name: str) -> bool:
        """Check if tool is a read operation."""
        
        if not tool_name:
            return False
        
        read_prefixes = ["get_", "list_", "read_", "fetch_", "query_"]
        return any(tool_name.startswith(p) for p in read_prefixes)
    
    def get_approval_mode(
        self,
        server_label: str,
        tool_name: str = None
    ) -> str:
        """Get approval mode for server/tool."""
        
        if self.is_trusted(server_label, tool_name):
            return "never"
        
        return "always"
    
    def get_server_config(self, label: str) -> Optional[dict]:
        """Get tool config for a trusted server."""
        
        if label not in self.trusted_servers:
            return None
        
        server = self.trusted_servers[label]
        
        return {
            "type": "mcp",
            "server_label": server.label,
            "server_url": server.url,
            "require_approval": self.get_approval_mode(label)
        }


# Usage
trust_manager = TrustManager()

# Add trusted servers
trust_manager.add_trusted_server(
    label="internal_weather",
    url="https://internal.corp/mcp/weather",
    trust_level="full",
    added_by="security_team"
)

trust_manager.add_trusted_server(
    label="external_api",
    url="https://api.external.com/mcp",
    trusted_tools=["get_data", "list_items"],
    trust_level="partial",
    added_by="admin"
)

# Check trust
print(trust_manager.is_trusted("internal_weather"))  # True
print(trust_manager.is_trusted("external_api", "get_data"))  # True
print(trust_manager.is_trusted("external_api", "delete_data"))  # False
```

---

## Complete Approval Workflow

### End-to-End Approval System

```python
import time
from typing import Callable

class ApprovalWorkflow:
    """Complete approval workflow system."""
    
    def __init__(
        self,
        trust_manager: TrustManager = None,
        approval_callback: Callable[[ApprovalRequest], bool] = None,
        timeout_seconds: int = 300
    ):
        self.client = OpenAI()
        self.trust_manager = trust_manager or TrustManager()
        self.approval_callback = approval_callback or self._default_callback
        self.timeout = timeout_seconds
        
        self.request_handler = ApprovalRequestHandler(timeout_seconds)
        self.response_builder = ApprovalResponseBuilder()
        
        self.workflow_history: List[dict] = []
    
    def _default_callback(self, request: ApprovalRequest) -> bool:
        """Default approval callback - interactive."""
        
        print("\n" + "=" * 50)
        print("APPROVAL REQUEST")
        print("=" * 50)
        print(f"Tool: {request.tool_name}")
        print(f"Server: {request.server_label}")
        print(f"Arguments: {request.arguments}")
        print(f"Expires: {request.expires_at}")
        
        response = input("\nApprove? (y/n): ").strip().lower()
        return response in ("y", "yes")
    
    def execute_with_approval(
        self,
        query: str,
        servers: List[dict]
    ) -> dict:
        """Execute query with approval handling."""
        
        # Apply trust settings to server configs
        configured_servers = []
        for server in servers:
            label = server.get("server_label", "")
            config = server.copy()
            
            # Override approval based on trust
            config["require_approval"] = self.trust_manager.get_approval_mode(label)
            configured_servers.append(config)
        
        # Initial request
        response = self.client.responses.create(
            model="gpt-4o",
            tools=configured_servers,
            input=query
        )
        
        # Check for approval requests
        pending = self.request_handler.process_response(response)
        
        if not pending:
            # No approval needed
            return {
                "success": True,
                "output": response.output_text,
                "approvals_required": 0
            }
        
        # Process each approval request
        approvals = []
        denials = []
        
        for request in pending:
            # Check trust first
            if self.trust_manager.is_trusted(request.server_label, request.tool_name):
                self.request_handler.approve(request.id, "Auto-approved (trusted)")
                approval = self.response_builder.approve(
                    request,
                    approved_by="trust_manager",
                    reason="Server is trusted"
                )
                approvals.append(approval)
                continue
            
            # Use callback for decision
            approved = self.approval_callback(request)
            
            if approved:
                self.request_handler.approve(request.id)
                approval = self.response_builder.approve(
                    request,
                    approved_by="user",
                    reason="User approved"
                )
                approvals.append(approval)
            else:
                self.request_handler.deny(request.id)
                denial = self.response_builder.deny(
                    request,
                    denied_by="user",
                    reason="User denied"
                )
                denials.append(denial)
        
        # Record history
        self.workflow_history.append({
            "query": query,
            "requests": len(pending),
            "approved": len(approvals),
            "denied": len(denials),
            "timestamp": datetime.now().isoformat()
        })
        
        # Submit approvals
        if approvals:
            result = self.response_builder.submit_batch(
                response.id,
                approvals
            )
            
            return {
                "success": True,
                "output": result["output"],
                "approvals_required": len(pending),
                "approved": len(approvals),
                "denied": len(denials)
            }
        
        # All denied
        return {
            "success": False,
            "output": "All tool calls were denied",
            "approvals_required": len(pending),
            "approved": 0,
            "denied": len(denials)
        }
    
    def get_statistics(self) -> dict:
        """Get approval workflow statistics."""
        
        if not self.workflow_history:
            return {"message": "No workflows executed"}
        
        total_requests = sum(w["requests"] for w in self.workflow_history)
        total_approved = sum(w["approved"] for w in self.workflow_history)
        total_denied = sum(w["denied"] for w in self.workflow_history)
        
        return {
            "total_workflows": len(self.workflow_history),
            "total_requests": total_requests,
            "total_approved": total_approved,
            "total_denied": total_denied,
            "approval_rate": (total_approved / total_requests * 100) if total_requests else 0
        }


# Usage
workflow = ApprovalWorkflow(
    trust_manager=trust_manager,
    timeout_seconds=300
)

# Execute with approval handling
# result = workflow.execute_with_approval(
#     "Delete all expired sessions from the database",
#     servers=[{
#         "type": "mcp",
#         "server_label": "database",
#         "server_url": "https://mcp.example.com/database"
#     }]
# )
```

---

## Hands-on Exercise

### Your Task

Build a complete MCP approval system with trust management.

### Requirements

1. Configure trusted servers
2. Handle approval requests
3. Submit approval responses
4. Track approval history

<details>
<summary>💡 Hints</summary>

- Use trust levels for different servers
- Track who approved what
- Handle timeouts gracefully
</details>

<details>
<summary>✅ Solution</summary>

```python
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
import json

class TrustLevel(Enum):
    NONE = "none"
    READ_ONLY = "read_only"
    PARTIAL = "partial"
    FULL = "full"


class RequestStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ApprovalAuditEntry:
    """Audit entry for approval decisions."""
    
    request_id: str
    tool_name: str
    server_label: str
    decision: str
    decided_by: str
    reason: Optional[str]
    timestamp: datetime
    auto_decision: bool


class ComprehensiveApprovalSystem:
    """Complete approval system with audit trail."""
    
    def __init__(
        self,
        timeout_seconds: int = 300,
        auto_approve_read_only: bool = True
    ):
        self.client = OpenAI()
        self.timeout = timeout_seconds
        self.auto_approve_read = auto_approve_read_only
        
        # Trust configuration
        self.trusted_servers: Dict[str, TrustLevel] = {}
        self.trusted_tools: Dict[str, Set[str]] = {}  # server -> set of tools
        
        # Request tracking
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        
        # Audit trail
        self.audit_log: List[ApprovalAuditEntry] = []
        
        # Approval rules
        self.approval_rules: List[Callable[[ApprovalRequest], Optional[bool]]] = []
    
    # Trust management
    
    def trust_server(
        self,
        label: str,
        level: TrustLevel,
        tools: List[str] = None
    ):
        """Add trusted server."""
        
        self.trusted_servers[label] = level
        
        if tools:
            self.trusted_tools[label] = set(tools)
    
    def revoke_trust(self, label: str):
        """Revoke server trust."""
        
        if label in self.trusted_servers:
            del self.trusted_servers[label]
        
        if label in self.trusted_tools:
            del self.trusted_tools[label]
    
    def check_trust(
        self,
        server_label: str,
        tool_name: str
    ) -> TrustLevel:
        """Check trust level for server/tool combination."""
        
        if server_label not in self.trusted_servers:
            return TrustLevel.NONE
        
        level = self.trusted_servers[server_label]
        
        # Check if tool is specifically trusted
        if server_label in self.trusted_tools:
            if tool_name not in self.trusted_tools[server_label]:
                # Tool not in trusted list
                if level == TrustLevel.PARTIAL:
                    return TrustLevel.NONE
        
        return level
    
    # Approval rules
    
    def add_rule(self, rule: Callable[[ApprovalRequest], Optional[bool]]):
        """Add approval rule. Return True/False for auto-decision, None for manual."""
        self.approval_rules.append(rule)
    
    def _evaluate_rules(self, request: ApprovalRequest) -> Optional[bool]:
        """Evaluate all rules for a request."""
        
        for rule in self.approval_rules:
            result = rule(request)
            if result is not None:
                return result
        
        return None
    
    # Request processing
    
    def process_response(self, response) -> List[ApprovalRequest]:
        """Extract and process approval requests."""
        
        requests = []
        now = datetime.now()
        
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'mcp_approval_request':
                request = ApprovalRequest(
                    id=item.id,
                    tool_name=item.name,
                    server_label=getattr(item, 'server_label', ''),
                    arguments=getattr(item, 'arguments', {}),
                    created_at=now,
                    expires_at=now + timedelta(seconds=self.timeout)
                )
                
                # Check for auto-decision
                auto_decision = self._auto_decide(request)
                
                if auto_decision is not None:
                    request.status = "auto_approved" if auto_decision else "denied"
                    self._log_decision(
                        request,
                        "approved" if auto_decision else "denied",
                        "system",
                        "Auto-decision based on trust/rules",
                        auto=True
                    )
                else:
                    self.pending_requests[request.id] = request
                
                requests.append(request)
        
        return requests
    
    def _auto_decide(self, request: ApprovalRequest) -> Optional[bool]:
        """Try to auto-decide based on trust and rules."""
        
        # Check trust level
        trust = self.check_trust(request.server_label, request.tool_name)
        
        if trust == TrustLevel.FULL:
            return True
        
        if trust == TrustLevel.READ_ONLY:
            if self._is_read_operation(request.tool_name):
                return True
        
        if trust == TrustLevel.PARTIAL:
            if request.server_label in self.trusted_tools:
                if request.tool_name in self.trusted_tools[request.server_label]:
                    return True
        
        # Evaluate custom rules
        return self._evaluate_rules(request)
    
    def _is_read_operation(self, tool_name: str) -> bool:
        """Check if tool is read-only."""
        read_prefixes = ["get_", "list_", "read_", "fetch_", "find_", "search_"]
        return any(tool_name.startswith(p) for p in read_prefixes)
    
    # Manual decisions
    
    def approve(
        self,
        request_id: str,
        approved_by: str,
        reason: str = None
    ) -> bool:
        """Manually approve a request."""
        
        if request_id not in self.pending_requests:
            return False
        
        request = self.pending_requests[request_id]
        
        if datetime.now() > request.expires_at:
            request.status = "expired"
            return False
        
        request.status = "approved"
        del self.pending_requests[request_id]
        
        self._log_decision(request, "approved", approved_by, reason, auto=False)
        return True
    
    def deny(
        self,
        request_id: str,
        denied_by: str,
        reason: str = None
    ) -> bool:
        """Manually deny a request."""
        
        if request_id not in self.pending_requests:
            return False
        
        request = self.pending_requests[request_id]
        request.status = "denied"
        del self.pending_requests[request_id]
        
        self._log_decision(request, "denied", denied_by, reason, auto=False)
        return True
    
    # Audit
    
    def _log_decision(
        self,
        request: ApprovalRequest,
        decision: str,
        decided_by: str,
        reason: Optional[str],
        auto: bool
    ):
        """Log an approval decision."""
        
        entry = ApprovalAuditEntry(
            request_id=request.id,
            tool_name=request.tool_name,
            server_label=request.server_label,
            decision=decision,
            decided_by=decided_by,
            reason=reason,
            timestamp=datetime.now(),
            auto_decision=auto
        )
        
        self.audit_log.append(entry)
    
    def get_audit_report(
        self,
        since: datetime = None,
        server_label: str = None
    ) -> dict:
        """Get audit report."""
        
        entries = self.audit_log
        
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        
        if server_label:
            entries = [e for e in entries if e.server_label == server_label]
        
        # Calculate statistics
        total = len(entries)
        approved = sum(1 for e in entries if e.decision == "approved")
        denied = sum(1 for e in entries if e.decision == "denied")
        auto_decisions = sum(1 for e in entries if e.auto_decision)
        
        by_server = {}
        by_tool = {}
        by_user = {}
        
        for entry in entries:
            # By server
            server = entry.server_label
            if server not in by_server:
                by_server[server] = {"approved": 0, "denied": 0}
            by_server[server][entry.decision] += 1
            
            # By tool
            tool = entry.tool_name
            if tool not in by_tool:
                by_tool[tool] = {"approved": 0, "denied": 0}
            by_tool[tool][entry.decision] += 1
            
            # By user
            user = entry.decided_by
            if user not in by_user:
                by_user[user] = 0
            by_user[user] += 1
        
        return {
            "period": {
                "from": since.isoformat() if since else "all time",
                "to": datetime.now().isoformat()
            },
            "summary": {
                "total_decisions": total,
                "approved": approved,
                "denied": denied,
                "approval_rate": (approved / total * 100) if total else 0,
                "auto_decisions": auto_decisions,
                "manual_decisions": total - auto_decisions
            },
            "by_server": by_server,
            "by_tool": by_tool,
            "by_user": by_user
        }
    
    # Response building
    
    def build_responses(
        self,
        requests: List[ApprovalRequest]
    ) -> List[dict]:
        """Build API responses for processed requests."""
        
        responses = []
        
        for request in requests:
            if request.status in ["approved", "auto_approved"]:
                responses.append({
                    "type": "mcp_approval_response",
                    "approval_id": request.id,
                    "approved": True
                })
            elif request.status == "denied":
                responses.append({
                    "type": "mcp_approval_response",
                    "approval_id": request.id,
                    "approved": False
                })
        
        return responses
    
    def export_audit_log(self, file_path: str = None) -> str:
        """Export audit log as JSON."""
        
        data = {
            "exported_at": datetime.now().isoformat(),
            "trust_config": {
                label: level.value
                for label, level in self.trusted_servers.items()
            },
            "audit_entries": [
                {
                    "request_id": e.request_id,
                    "tool": e.tool_name,
                    "server": e.server_label,
                    "decision": e.decision,
                    "decided_by": e.decided_by,
                    "reason": e.reason,
                    "timestamp": e.timestamp.isoformat(),
                    "auto": e.auto_decision
                }
                for e in self.audit_log
            ]
        }
        
        json_str = json.dumps(data, indent=2)
        
        if file_path:
            with open(file_path, "w") as f:
                f.write(json_str)
        
        return json_str


# Usage example
system = ComprehensiveApprovalSystem(
    timeout_seconds=300,
    auto_approve_read_only=True
)

# Configure trusted servers
system.trust_server("internal_api", TrustLevel.FULL)
system.trust_server("external_data", TrustLevel.READ_ONLY)
system.trust_server("database", TrustLevel.PARTIAL, ["query", "get_schema"])

# Add custom rule
system.add_rule(lambda req: 
    False if "delete" in req.tool_name.lower() else None
)

# Process requests
# response = client.responses.create(...)
# requests = system.process_response(response)
# 
# # Handle pending
# for req in system.pending_requests.values():
#     # Get user decision
#     approved = input(f"Approve {req.tool_name}? (y/n): ") == 'y'
#     
#     if approved:
#         system.approve(req.id, "admin", "Approved by admin")
#     else:
#         system.deny(req.id, "admin", "Denied by admin")

# Get audit report
report = system.get_audit_report()
print(f"Total decisions: {report['summary']['total_decisions']}")
print(f"Approval rate: {report['summary']['approval_rate']:.1f}%")

# Export log
# system.export_audit_log("approval_audit.json")
```

</details>

---

## Summary

✅ require_approval controls when human input is needed  
✅ mcp_approval_request contains tool and argument details  
✅ mcp_approval_response submits the decision  
✅ Trusted servers can bypass approval requirements  
✅ Audit trails track all approval decisions  
✅ Custom rules enable automated approval logic

**Next:** [OpenAI Connectors](./07-openai-connectors.md)

---

## Further Reading

- [MCP Approval Documentation](https://platform.openai.com/docs/guides/mcp#approval) — Official guide
- [Security Best Practices](https://platform.openai.com/docs/guides/mcp#security) — Security considerations
- [Trust Configuration](https://modelcontextprotocol.io/security) — MCP security model
