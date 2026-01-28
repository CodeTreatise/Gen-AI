---
title: "Encrypted Reasoning"
---

# Encrypted Reasoning

## Introduction

For organizations with Zero Data Retention (ZDR) requirements, reasoning models support encrypted reasoning. This feature allows ZDR organizations to benefit from multi-turn reasoning capabilities while maintaining their data retention policies through encrypted, in-memory reasoning content.

### What We'll Cover

- Understanding ZDR and encrypted reasoning
- Working with encrypted reasoning items
- In-memory decryption behavior
- Stateless mode with reasoning benefits

### Prerequisites

- Multi-turn reasoning concepts
- Understanding of data retention policies
- API authentication and organization settings

---

## Understanding Encrypted Reasoning

### What is Zero Data Retention (ZDR)?

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class RetentionPolicy(str, Enum):
    """Data retention policies."""
    
    STANDARD = "standard"  # Data may be retained
    ZERO_DATA_RETENTION = "zdr"  # No data retained


@dataclass
class ZDRFeature:
    """Feature of Zero Data Retention."""
    
    feature: str
    description: str
    behavior: str


ZDR_FEATURES = [
    ZDRFeature(
        feature="No data storage",
        description="Prompts and completions are not stored",
        behavior="Data processed in memory only"
    ),
    ZDRFeature(
        feature="No training data",
        description="Data not used for model training",
        behavior="Complete data isolation"
    ),
    ZDRFeature(
        feature="Encrypted reasoning",
        description="Reasoning tokens encrypted when returned",
        behavior="Decrypted only during processing"
    ),
    ZDRFeature(
        feature="Multi-turn support",
        description="Reasoning preserved across turns via encryption",
        behavior="Stateless but coherent conversations"
    )
]


print("Zero Data Retention (ZDR) Features")
print("=" * 60)

for feature in ZDR_FEATURES:
    print(f"\n🔒 {feature.feature}")
    print(f"   {feature.description}")
    print(f"   → {feature.behavior}")


print("""

📊 Standard vs ZDR Organization Comparison

┌────────────────────┬────────────────┬────────────────┐
│ Feature            │ Standard       │ ZDR            │
├────────────────────┼────────────────┼────────────────┤
│ Data retention     │ Yes (30 days)  │ No             │
│ Reasoning visible  │ Via summaries  │ Via summaries  │
│ Multi-turn context │ Stored         │ Encrypted      │
│ Model training     │ Opt-out avail. │ Never          │
│ Audit logging      │ Full           │ Metadata only  │
└────────────────────┴────────────────┴────────────────┘
""")
```

### How Encrypted Reasoning Works

```python
@dataclass
class EncryptedReasoningItem:
    """An encrypted reasoning item."""
    
    item_type: str = "reasoning"
    item_id: str = ""
    encrypted_content: str = ""  # Base64 encoded encrypted data
    is_encrypted: bool = True


@dataclass
class ReasoningEncryptionFlow:
    """Flow of encrypted reasoning."""
    
    step: int
    stage: str
    action: str
    data_state: str


ENCRYPTION_FLOW = [
    ReasoningEncryptionFlow(
        step=1,
        stage="Generation",
        action="Model generates reasoning tokens",
        data_state="Plaintext (in memory)"
    ),
    ReasoningEncryptionFlow(
        step=2,
        stage="Encryption",
        action="Reasoning encrypted before response",
        data_state="Encrypted"
    ),
    ReasoningEncryptionFlow(
        step=3,
        stage="Response",
        action="Encrypted reasoning returned to client",
        data_state="Encrypted (opaque to client)"
    ),
    ReasoningEncryptionFlow(
        step=4,
        stage="Storage (client)",
        action="Client stores encrypted item",
        data_state="Encrypted (client cannot read)"
    ),
    ReasoningEncryptionFlow(
        step=5,
        stage="Next request",
        action="Client sends encrypted item back",
        data_state="Encrypted"
    ),
    ReasoningEncryptionFlow(
        step=6,
        stage="Decryption",
        action="API decrypts for model processing",
        data_state="Plaintext (in memory only)"
    ),
    ReasoningEncryptionFlow(
        step=7,
        stage="Processing",
        action="Model uses previous reasoning",
        data_state="Plaintext (in memory)"
    ),
    ReasoningEncryptionFlow(
        step=8,
        stage="Cleanup",
        action="All plaintext discarded after response",
        data_state="No retention"
    )
]


print("\n\nEncrypted Reasoning Flow")
print("=" * 60)

for flow in ENCRYPTION_FLOW:
    print(f"\n{flow.step}. {flow.stage}")
    print(f"   Action: {flow.action}")
    print(f"   Data: {flow.data_state}")


print("""

🔐 Encryption Diagram

Request 1:
┌─────────────┐     ┌─────────────────────────┐
│ User Input  │ ──▶ │ Model Processing        │
└─────────────┘     │ (reasoning in memory)   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ ENCRYPT reasoning       │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ Return encrypted item   │
                    │ + visible response      │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ All plaintext DELETED   │
                    └─────────────────────────┘

Request 2:
┌─────────────┐     ┌─────────────────────────┐
│ User Input  │     │ Previous encrypted      │
│             │ + ─▶│ reasoning item          │
└─────────────┘     └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ DECRYPT (in memory)     │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ Model uses prev.        │
                    │ reasoning + new input   │
                    └─────────────────────────┘
""")
```

---

## Working with Encrypted Reasoning Items

### Requesting Encrypted Content

```python
from typing import Dict, Any


class ZDRReasoningClient:
    """Client for ZDR organizations using encrypted reasoning."""
    
    def __init__(self, model: str = "gpt-5"):
        self.model = model
        self.encrypted_items: List[Dict[str, Any]] = []
    
    def create_request(
        self,
        messages: List[dict],
        include_encrypted: bool = True
    ) -> dict:
        """Create a request that includes encrypted reasoning."""
        
        # Build input
        input_items = []
        
        # Add messages
        for msg in messages:
            input_items.append({
                "type": "message",
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add previously received encrypted reasoning
        if include_encrypted:
            input_items.extend(self.encrypted_items)
        
        # Build request with include parameter
        request = {
            "model": self.model,
            "input": input_items
        }
        
        if include_encrypted:
            request["include"] = ["reasoning.encrypted_content"]
        
        return request
    
    def extract_encrypted_items(self, response: dict) -> List[dict]:
        """Extract encrypted reasoning items from response."""
        
        encrypted = []
        
        for item in response.get("output", []):
            if item.get("type") == "reasoning":
                # Check for encrypted content
                if "encrypted_content" in item:
                    encrypted.append({
                        "type": "reasoning",
                        "id": item.get("id", ""),
                        "encrypted_content": item["encrypted_content"]
                    })
        
        return encrypted
    
    def update_state(self, response: dict):
        """Update client state with new encrypted items."""
        
        # Replace previous encrypted items with new ones
        self.encrypted_items = self.extract_encrypted_items(response)
    
    def get_state(self) -> dict:
        """Get current state."""
        
        return {
            "has_encrypted_reasoning": len(self.encrypted_items) > 0,
            "encrypted_item_count": len(self.encrypted_items),
            "encrypted_ids": [item.get("id") for item in self.encrypted_items]
        }


print("\nZDR Reasoning Client")
print("=" * 60)

client = ZDRReasoningClient("gpt-5")

# First request
request1 = client.create_request([
    {"role": "user", "content": "Analyze this complex problem..."}
])

print("\n📤 Turn 1 Request:")
print(f"   Model: {request1['model']}")
print(f"   Include: {request1.get('include', [])}")
print(f"   Encrypted items passed: {len(client.encrypted_items)}")

# Simulate response with encrypted content
mock_response1 = {
    "output": [
        {
            "type": "reasoning",
            "id": "enc_reason_001",
            "encrypted_content": "gAAAAABl...encrypted_base64_content..."
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Here's my analysis..."}]
        }
    ]
}

# Update state
client.update_state(mock_response1)

print(f"\n📥 Turn 1 Response received")
print(f"   State: {client.get_state()}")

# Second request (with encrypted context)
request2 = client.create_request([
    {"role": "user", "content": "Analyze this complex problem..."},
    {"role": "assistant", "content": "Here's my analysis..."},
    {"role": "user", "content": "Can you elaborate on point 2?"}
])

print(f"\n📤 Turn 2 Request:")
print(f"   Include: {request2.get('include', [])}")
print(f"   Encrypted items passed: {len(client.encrypted_items)}")
print("   ✅ Previous reasoning will be decrypted for processing")
```

### Validating Encrypted Items

```python
import base64
from typing import Tuple


class EncryptedItemValidator:
    """Validate encrypted reasoning items."""
    
    @staticmethod
    def validate_item(item: dict) -> Tuple[bool, List[str]]:
        """Validate an encrypted reasoning item."""
        
        errors = []
        
        # Check required fields
        if item.get("type") != "reasoning":
            errors.append("Item type must be 'reasoning'")
        
        if "encrypted_content" not in item:
            errors.append("Missing 'encrypted_content' field")
        
        if "id" not in item:
            errors.append("Missing 'id' field")
        
        # Validate encrypted content format
        encrypted = item.get("encrypted_content", "")
        if encrypted:
            try:
                # Check if it's valid base64
                decoded = base64.b64decode(encrypted)
                if len(decoded) < 16:  # Too short to be valid encryption
                    errors.append("Encrypted content too short")
            except Exception:
                errors.append("Invalid base64 encoding in encrypted_content")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_for_request(items: List[dict]) -> dict:
        """Validate items before including in request."""
        
        valid_items = []
        invalid_items = []
        
        for item in items:
            is_valid, errors = EncryptedItemValidator.validate_item(item)
            
            if is_valid:
                valid_items.append(item)
            else:
                invalid_items.append({
                    "item": item,
                    "errors": errors
                })
        
        return {
            "valid": valid_items,
            "invalid": invalid_items,
            "all_valid": len(invalid_items) == 0,
            "summary": f"{len(valid_items)} valid, {len(invalid_items)} invalid"
        }


print("\n\nEncrypted Item Validation")
print("=" * 60)

validator = EncryptedItemValidator()

# Test items
test_items = [
    {
        "type": "reasoning",
        "id": "enc_001",
        "encrypted_content": base64.b64encode(b"x" * 128).decode()
    },
    {
        "type": "reasoning",
        "id": "enc_002",
        "encrypted_content": "not-valid-base64!!"
    },
    {
        "type": "message",  # Wrong type
        "id": "enc_003"
    }
]

result = validator.validate_for_request(test_items)

print(f"\n📊 Validation Result: {result['summary']}")
print(f"   All valid: {result['all_valid']}")

if result['invalid']:
    print("\n❌ Invalid items:")
    for invalid in result['invalid']:
        print(f"   ID: {invalid['item'].get('id', 'unknown')}")
        for error in invalid['errors']:
            print(f"      • {error}")
```

---

## Stateless Mode with Reasoning Benefits

### Understanding Stateless + Encrypted

```python
@dataclass
class StatelessBenefit:
    """Benefit of stateless mode with encrypted reasoning."""
    
    benefit: str
    without_encryption: str
    with_encryption: str


STATELESS_BENEFITS = [
    StatelessBenefit(
        benefit="Multi-turn coherence",
        without_encryption="Model re-analyzes from scratch each turn",
        with_encryption="Model builds on previous reasoning"
    ),
    StatelessBenefit(
        benefit="Cost efficiency",
        without_encryption="Full reasoning repeated each turn",
        with_encryption="Reasoning continued, not repeated"
    ),
    StatelessBenefit(
        benefit="Response quality",
        without_encryption="May contradict previous responses",
        with_encryption="Consistent with previous analysis"
    ),
    StatelessBenefit(
        benefit="Data privacy",
        without_encryption="All data sent in plaintext each time",
        with_encryption="Reasoning opaque to client"
    ),
    StatelessBenefit(
        benefit="Compliance",
        without_encryption="Client must ensure data handling",
        with_encryption="Server handles encryption/retention"
    )
]


print("Stateless Mode: With vs Without Encrypted Reasoning")
print("=" * 60)

for benefit in STATELESS_BENEFITS:
    print(f"\n✅ {benefit.benefit}")
    print(f"   Without: {benefit.without_encryption}")
    print(f"   With:    {benefit.with_encryption}")
```

### Building a Stateless ZDR System

```python
from datetime import datetime


@dataclass
class ZDRSession:
    """A ZDR-compliant session with encrypted reasoning."""
    
    session_id: str
    created_at: datetime
    messages: List[dict]
    encrypted_reasoning: List[dict]
    turn_count: int = 0
    
    def add_message(self, role: str, content: str):
        """Add a message to the session."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def update_reasoning(self, items: List[dict]):
        """Update encrypted reasoning (replaces previous)."""
        self.encrypted_reasoning = items
        self.turn_count += 1


class ZDRConversationSystem:
    """Stateless conversation system for ZDR organizations."""
    
    def __init__(self, model: str = "gpt-5"):
        self.model = model
        self.validator = EncryptedItemValidator()
    
    def create_session(self, session_id: str) -> ZDRSession:
        """Create a new ZDR session."""
        
        return ZDRSession(
            session_id=session_id,
            created_at=datetime.now(),
            messages=[],
            encrypted_reasoning=[]
        )
    
    def prepare_request(
        self,
        session: ZDRSession,
        user_message: str
    ) -> dict:
        """Prepare a request for a ZDR session."""
        
        # Add user message to session
        session.add_message("user", user_message)
        
        # Build input items
        input_items = []
        
        # Add all messages
        for msg in session.messages:
            input_items.append({
                "type": "message",
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add encrypted reasoning (if any)
        if session.encrypted_reasoning:
            validation = self.validator.validate_for_request(
                session.encrypted_reasoning
            )
            
            if validation["all_valid"]:
                input_items.extend(validation["valid"])
        
        return {
            "model": self.model,
            "input": input_items,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "medium"}
        }
    
    def process_response(
        self,
        session: ZDRSession,
        response: dict
    ) -> str:
        """Process response and update session."""
        
        output_text = ""
        encrypted_items = []
        
        for item in response.get("output", []):
            item_type = item.get("type")
            
            if item_type == "reasoning":
                if "encrypted_content" in item:
                    encrypted_items.append({
                        "type": "reasoning",
                        "id": item.get("id"),
                        "encrypted_content": item["encrypted_content"]
                    })
            
            elif item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        output_text = part.get("text", "")
        
        # Update session
        session.add_message("assistant", output_text)
        session.update_reasoning(encrypted_items)
        
        return output_text
    
    def get_session_info(self, session: ZDRSession) -> dict:
        """Get session information (ZDR compliant)."""
        
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "turn_count": session.turn_count,
            "message_count": len(session.messages),
            "has_encrypted_reasoning": len(session.encrypted_reasoning) > 0,
            # Note: We don't expose encrypted content details
            "zdr_compliant": True
        }


print("\n\nZDR Conversation System")
print("=" * 60)

system = ZDRConversationSystem("gpt-5")
session = system.create_session("zdr_session_001")

print(f"\n📝 Session created: {session.session_id}")
print(f"   ZDR compliant: ✅")

# Prepare first request
request = system.prepare_request(session, "Analyze this confidential data...")

print(f"\n📤 Request prepared:")
print(f"   Model: {request['model']}")
print(f"   Include: {request['include']}")
print(f"   Input items: {len(request['input'])}")

# Simulate response
mock_response = {
    "output": [
        {
            "type": "reasoning",
            "id": "zdr_reason_001",
            "encrypted_content": "gAAAAABl_encrypted_reasoning_content_here..."
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Here's my analysis of the data..."}]
        }
    ]
}

output = system.process_response(session, mock_response)

print(f"\n📥 Response: {output[:50]}...")
print(f"\n📊 Session Info:")
for key, value in system.get_session_info(session).items():
    print(f"   {key}: {value}")
```

---

## Security Considerations

### Best Practices for Encrypted Reasoning

```python
SECURITY_GUIDELINES = [
    {
        "guideline": "Never attempt to decrypt client-side",
        "reason": "Encrypted content is server-key encrypted",
        "implementation": "Store encrypted items as opaque blobs"
    },
    {
        "guideline": "Don't log encrypted content",
        "reason": "May contain sensitive reasoning",
        "implementation": "Log only item IDs, not content"
    },
    {
        "guideline": "Set session timeouts",
        "reason": "Encrypted items should not persist indefinitely",
        "implementation": "Clear encrypted items after session end"
    },
    {
        "guideline": "Validate before sending",
        "reason": "Corrupted items will cause errors",
        "implementation": "Check base64 encoding and structure"
    },
    {
        "guideline": "Use HTTPS only",
        "reason": "Prevent interception of encrypted content",
        "implementation": "Enforce TLS for all API calls"
    },
    {
        "guideline": "Minimize retention",
        "reason": "ZDR compliance means minimal data holding",
        "implementation": "Clear session data promptly after use"
    }
]


print("Security Guidelines for Encrypted Reasoning")
print("=" * 60)

for i, guideline in enumerate(SECURITY_GUIDELINES, 1):
    print(f"\n{i}. {guideline['guideline']}")
    print(f"   Reason: {guideline['reason']}")
    print(f"   How: {guideline['implementation']}")


class SecureZDRHandler:
    """Secure handler for ZDR encrypted reasoning."""
    
    def __init__(self):
        self.session_timeout_minutes = 30
    
    def store_encrypted_safely(
        self,
        session_id: str,
        encrypted_items: List[dict]
    ) -> dict:
        """Store encrypted items with security measures."""
        
        # Extract only necessary fields
        safe_items = []
        for item in encrypted_items:
            safe_items.append({
                "type": "reasoning",
                "id": item.get("id", ""),
                "encrypted_content": item.get("encrypted_content", "")
                # Don't store anything else
            })
        
        return {
            "session_id": session_id,
            "item_count": len(safe_items),
            "items": safe_items,
            "stored_at": datetime.now().isoformat(),
            "expires_at": (
                datetime.now() + 
                __import__('datetime').timedelta(minutes=self.session_timeout_minutes)
            ).isoformat()
        }
    
    def log_safely(
        self,
        session_id: str,
        encrypted_items: List[dict],
        action: str
    ) -> str:
        """Create a safe log entry (no sensitive data)."""
        
        # Log only metadata, never content
        return (
            f"[{datetime.now().isoformat()}] "
            f"Session: {session_id}, "
            f"Action: {action}, "
            f"Items: {len(encrypted_items)}, "
            f"IDs: {[item.get('id') for item in encrypted_items]}"
        )
    
    def clear_session(self, session_id: str) -> dict:
        """Securely clear session data."""
        
        return {
            "session_id": session_id,
            "action": "cleared",
            "timestamp": datetime.now().isoformat(),
            "message": "All encrypted reasoning items discarded"
        }


print("\n\nSecure ZDR Handler Demo")
print("=" * 60)

handler = SecureZDRHandler()

# Demo secure storage
sample_encrypted = [
    {"type": "reasoning", "id": "r1", "encrypted_content": "abc123..."}
]

stored = handler.store_encrypted_safely("session_001", sample_encrypted)
print(f"\n🔒 Secure storage:")
print(f"   Items stored: {stored['item_count']}")
print(f"   Expires: {stored['expires_at']}")

# Demo safe logging
log_entry = handler.log_safely("session_001", sample_encrypted, "received")
print(f"\n📋 Safe log entry:")
print(f"   {log_entry}")

# Demo session clearing
cleared = handler.clear_session("session_001")
print(f"\n🗑️  Session cleared:")
print(f"   {cleared['message']}")
```

---

## Hands-on Exercise

### Your Task

Build a complete ZDR-compliant conversation system that properly handles encrypted reasoning, enforces security guidelines, and provides audit-friendly logging.

### Requirements

1. Manage sessions with encrypted reasoning preservation
2. Implement security best practices
3. Create audit-friendly (but safe) logging
4. Handle session expiration

<details>
<summary>💡 Hints</summary>

- Store only opaque encrypted blobs
- Log metadata, never content
- Set and enforce session timeouts
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib


class AuditAction(str, Enum):
    """Audit log action types."""
    
    SESSION_CREATED = "session_created"
    MESSAGE_SENT = "message_sent"
    RESPONSE_RECEIVED = "response_received"
    REASONING_STORED = "reasoning_stored"
    REASONING_USED = "reasoning_used"
    SESSION_EXPIRED = "session_expired"
    SESSION_CLEARED = "session_cleared"


@dataclass
class AuditEntry:
    """An audit log entry (ZDR compliant)."""
    
    timestamp: datetime
    session_id: str
    action: AuditAction
    metadata: Dict[str, Any]
    # Never includes actual content
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "action": self.action.value,
            "metadata": self.metadata
        }


@dataclass
class SecureSession:
    """A ZDR-compliant secure session."""
    
    session_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    message_hashes: List[str]  # Store hashes, not content
    encrypted_reasoning: List[dict]
    turn_count: int = 0
    is_active: bool = True
    
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


class ZDRComplianceSystem:
    """Complete ZDR-compliant conversation system."""
    
    def __init__(
        self,
        model: str = "gpt-5",
        session_timeout_minutes: int = 30,
        max_session_turns: int = 100
    ):
        self.model = model
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_turns = max_session_turns
        
        self.sessions: Dict[str, SecureSession] = {}
        self.audit_log: List[AuditEntry] = []
        
        # For reconstructing messages in requests
        # In production, use secure temporary storage
        self._temp_messages: Dict[str, List[dict]] = {}
    
    def _audit(
        self,
        session_id: str,
        action: AuditAction,
        **metadata
    ):
        """Create an audit entry."""
        
        entry = AuditEntry(
            timestamp=datetime.now(),
            session_id=session_id,
            action=action,
            metadata=metadata
        )
        
        self.audit_log.append(entry)
    
    def _hash_content(self, content: str) -> str:
        """Hash content for audit trail (not the actual content)."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def create_session(self, session_id: str) -> dict:
        """Create a new ZDR-compliant session."""
        
        now = datetime.now()
        
        session = SecureSession(
            session_id=session_id,
            created_at=now,
            last_activity=now,
            expires_at=now + self.session_timeout,
            message_hashes=[],
            encrypted_reasoning=[]
        )
        
        self.sessions[session_id] = session
        self._temp_messages[session_id] = []
        
        self._audit(
            session_id,
            AuditAction.SESSION_CREATED,
            timeout_minutes=self.session_timeout.total_seconds() / 60
        )
        
        return {
            "session_id": session_id,
            "created": True,
            "expires_at": session.expires_at.isoformat()
        }
    
    def _check_session(self, session_id: str) -> Optional[str]:
        """Check session validity, return error message or None."""
        
        if session_id not in self.sessions:
            return "Session not found"
        
        session = self.sessions[session_id]
        
        if session.is_expired():
            self._expire_session(session_id)
            return "Session expired"
        
        if not session.is_active:
            return "Session inactive"
        
        if session.turn_count >= self.max_turns:
            return "Maximum turns reached"
        
        return None
    
    def _expire_session(self, session_id: str):
        """Expire and clean up a session."""
        
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.is_active = False
            session.encrypted_reasoning = []  # Clear encrypted data
            
            self._audit(
                session_id,
                AuditAction.SESSION_EXPIRED,
                turn_count=session.turn_count
            )
        
        if session_id in self._temp_messages:
            del self._temp_messages[session_id]
    
    def send_message(
        self,
        session_id: str,
        user_message: str
    ) -> dict:
        """Send a message in a ZDR session."""
        
        # Check session
        error = self._check_session(session_id)
        if error:
            return {"error": error}
        
        session = self.sessions[session_id]
        
        # Store hash only
        content_hash = self._hash_content(user_message)
        session.message_hashes.append(f"user:{content_hash}")
        
        # Store actual message in temp storage (for request building)
        self._temp_messages[session_id].append({
            "role": "user",
            "content": user_message
        })
        
        # Update activity
        session.last_activity = datetime.now()
        session.expires_at = datetime.now() + self.session_timeout
        
        self._audit(
            session_id,
            AuditAction.MESSAGE_SENT,
            role="user",
            content_hash=content_hash,
            message_length=len(user_message)
        )
        
        # Build request
        input_items = []
        
        # Add messages
        for msg in self._temp_messages[session_id]:
            input_items.append({
                "type": "message",
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add encrypted reasoning
        if session.encrypted_reasoning:
            input_items.extend(session.encrypted_reasoning)
            
            self._audit(
                session_id,
                AuditAction.REASONING_USED,
                item_count=len(session.encrypted_reasoning)
            )
        
        return {
            "request": {
                "model": self.model,
                "input": input_items,
                "include": ["reasoning.encrypted_content"],
                "reasoning": {"effort": "medium"}
            }
        }
    
    def receive_response(
        self,
        session_id: str,
        response: dict
    ) -> dict:
        """Process response from API."""
        
        error = self._check_session(session_id)
        if error:
            return {"error": error}
        
        session = self.sessions[session_id]
        
        output_text = ""
        encrypted_items = []
        
        for item in response.get("output", []):
            item_type = item.get("type")
            
            if item_type == "reasoning":
                if "encrypted_content" in item:
                    # Store only necessary fields
                    encrypted_items.append({
                        "type": "reasoning",
                        "id": item.get("id"),
                        "encrypted_content": item["encrypted_content"]
                    })
            
            elif item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        output_text = part.get("text", "")
        
        # Update session
        if output_text:
            content_hash = self._hash_content(output_text)
            session.message_hashes.append(f"assistant:{content_hash}")
            
            self._temp_messages[session_id].append({
                "role": "assistant",
                "content": output_text
            })
        
        session.encrypted_reasoning = encrypted_items
        session.turn_count += 1
        session.last_activity = datetime.now()
        session.expires_at = datetime.now() + self.session_timeout
        
        self._audit(
            session_id,
            AuditAction.RESPONSE_RECEIVED,
            role="assistant",
            content_hash=self._hash_content(output_text),
            response_length=len(output_text)
        )
        
        if encrypted_items:
            self._audit(
                session_id,
                AuditAction.REASONING_STORED,
                item_count=len(encrypted_items),
                item_ids=[item.get("id") for item in encrypted_items]
            )
        
        return {
            "content": output_text,
            "reasoning_preserved": len(encrypted_items) > 0,
            "turn": session.turn_count
        }
    
    def clear_session(self, session_id: str) -> dict:
        """Clear a session and all its data."""
        
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        turn_count = session.turn_count
        
        # Clear all data
        session.encrypted_reasoning = []
        session.message_hashes = []
        session.is_active = False
        
        if session_id in self._temp_messages:
            del self._temp_messages[session_id]
        
        self._audit(
            session_id,
            AuditAction.SESSION_CLEARED,
            turns_completed=turn_count
        )
        
        return {
            "cleared": True,
            "session_id": session_id,
            "turns_completed": turn_count
        }
    
    def get_audit_log(
        self,
        session_id: Optional[str] = None,
        action: Optional[AuditAction] = None
    ) -> List[dict]:
        """Get audit log entries (ZDR compliant - no content)."""
        
        entries = self.audit_log
        
        if session_id:
            entries = [e for e in entries if e.session_id == session_id]
        
        if action:
            entries = [e for e in entries if e.action == action]
        
        return [e.to_dict() for e in entries]
    
    def cleanup_expired(self) -> dict:
        """Clean up all expired sessions."""
        
        expired = []
        
        for session_id, session in list(self.sessions.items()):
            if session.is_expired():
                self._expire_session(session_id)
                expired.append(session_id)
        
        return {
            "cleaned_up": len(expired),
            "session_ids": expired
        }


# Demo
print("\nZDR Compliance System Demo")
print("=" * 60)

system = ZDRComplianceSystem(
    model="gpt-5",
    session_timeout_minutes=30
)

# Create session
result = system.create_session("zdr_demo_001")
print(f"\n✅ Session: {result}")

# Send message
result = system.send_message(
    "zdr_demo_001",
    "Analyze this confidential financial data..."
)
print(f"\n📤 Message sent, request prepared")

# Simulate response
mock_response = {
    "output": [
        {
            "type": "reasoning",
            "id": "zdr_r1",
            "encrypted_content": "gAAAA_encrypted..."
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Analysis complete..."}]
        }
    ]
}

result = system.receive_response("zdr_demo_001", mock_response)
print(f"\n📥 Response: {result}")

# Check audit log
print(f"\n📋 Audit Log (ZDR Compliant):")
for entry in system.get_audit_log("zdr_demo_001"):
    print(f"   [{entry['action']}] {entry['metadata']}")

# Clear session
result = system.clear_session("zdr_demo_001")
print(f"\n🗑️  Session cleared: {result}")
```

</details>

---

## Summary

✅ ZDR organizations get encrypted reasoning for multi-turn support  
✅ Use `include: ["reasoning.encrypted_content"]` to receive encrypted items  
✅ Encrypted content is decrypted server-side during processing only  
✅ Store encrypted items as opaque blobs—never attempt client-side decryption  
✅ Implement session timeouts and audit logging for compliance

**Next:** [Reasoning Summaries](./06-reasoning-summaries.md)

---

## Further Reading

- [OpenAI Zero Data Retention](https://platform.openai.com/docs/models/default-usage-policies) — ZDR policies
- [Enterprise Privacy](https://openai.com/enterprise-privacy) — Enterprise features
- [API Security](https://platform.openai.com/docs/guides/safety-best-practices) — Best practices
