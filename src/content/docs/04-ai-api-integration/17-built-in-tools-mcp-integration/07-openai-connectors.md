---
title: "OpenAI Connectors"
---

# OpenAI Connectors

## Introduction

OpenAI Connectors provide pre-built integrations with popular services like Google Workspace, Microsoft 365, and Dropbox. These connectors handle OAuth authentication and provide access to calendars, email, files, and more.

### What We'll Cover

- Available connectors
- OAuth authorization setup
- connector_id configuration
- Using connectors in requests
- Managing connector lifecycles

### Prerequisites

- OpenAI API access
- Understanding of OAuth
- Service-specific account access

---

## Available Connectors

### Supported Services

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class ConnectorType(Enum):
    GOOGLE_CALENDAR = "google_calendar"
    GOOGLE_DRIVE = "google_drive"
    GMAIL = "gmail"
    MICROSOFT_OUTLOOK = "microsoft_outlook"
    MICROSOFT_TEAMS = "microsoft_teams"
    MICROSOFT_SHAREPOINT = "microsoft_sharepoint"
    DROPBOX = "dropbox"


@dataclass
class ConnectorCapabilities:
    """Capabilities of a connector."""
    
    connector_type: ConnectorType
    read_operations: List[str]
    write_operations: List[str]
    search_supported: bool
    oauth_scopes: List[str]


CONNECTOR_CAPABILITIES = {
    ConnectorType.GOOGLE_CALENDAR: ConnectorCapabilities(
        connector_type=ConnectorType.GOOGLE_CALENDAR,
        read_operations=["list_events", "get_event", "list_calendars"],
        write_operations=["create_event", "update_event", "delete_event"],
        search_supported=True,
        oauth_scopes=[
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/calendar.events"
        ]
    ),
    ConnectorType.GMAIL: ConnectorCapabilities(
        connector_type=ConnectorType.GMAIL,
        read_operations=["list_messages", "get_message", "list_labels"],
        write_operations=["send_message", "create_draft", "modify_labels"],
        search_supported=True,
        oauth_scopes=[
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send"
        ]
    ),
    ConnectorType.GOOGLE_DRIVE: ConnectorCapabilities(
        connector_type=ConnectorType.GOOGLE_DRIVE,
        read_operations=["list_files", "get_file", "download_file"],
        write_operations=["upload_file", "create_folder", "delete_file"],
        search_supported=True,
        oauth_scopes=[
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.file"
        ]
    ),
    ConnectorType.MICROSOFT_OUTLOOK: ConnectorCapabilities(
        connector_type=ConnectorType.MICROSOFT_OUTLOOK,
        read_operations=["list_messages", "get_message", "list_folders"],
        write_operations=["send_message", "create_draft", "move_message"],
        search_supported=True,
        oauth_scopes=[
            "Mail.Read",
            "Mail.Send"
        ]
    ),
    ConnectorType.DROPBOX: ConnectorCapabilities(
        connector_type=ConnectorType.DROPBOX,
        read_operations=["list_folder", "get_metadata", "download_file"],
        write_operations=["upload_file", "create_folder", "delete_file"],
        search_supported=True,
        oauth_scopes=[
            "files.metadata.read",
            "files.content.read",
            "files.content.write"
        ]
    )
}


def get_connector_info(connector_type: ConnectorType) -> dict:
    """Get information about a connector."""
    
    caps = CONNECTOR_CAPABILITIES.get(connector_type)
    if not caps:
        return {"error": "Unknown connector type"}
    
    return {
        "type": connector_type.value,
        "read_operations": caps.read_operations,
        "write_operations": caps.write_operations,
        "search_supported": caps.search_supported,
        "required_scopes": caps.oauth_scopes
    }


# Usage
info = get_connector_info(ConnectorType.GOOGLE_CALENDAR)
print(f"Calendar operations: {info['read_operations'] + info['write_operations']}")
```

---

## OAuth Authorization Setup

### Authorization Flow

```python
from openai import OpenAI
from urllib.parse import urlencode

client = OpenAI()


class OAuthFlowManager:
    """Manage OAuth authorization for connectors."""
    
    def __init__(self):
        self.client = OpenAI()
        self.pending_authorizations: Dict[str, dict] = {}
    
    def initiate_authorization(
        self,
        connector_type: ConnectorType,
        redirect_uri: str,
        user_id: str = None
    ) -> dict:
        """Start OAuth authorization flow."""
        
        # Create connector with OAuth
        connector = self.client.connectors.create(
            type=connector_type.value,
            oauth={
                "redirect_uri": redirect_uri
            }
        )
        
        # Store pending authorization
        self.pending_authorizations[connector.id] = {
            "connector_id": connector.id,
            "type": connector_type.value,
            "user_id": user_id,
            "status": "pending",
            "authorization_url": connector.authorization_url
        }
        
        return {
            "connector_id": connector.id,
            "authorization_url": connector.authorization_url,
            "message": "Redirect user to authorization URL"
        }
    
    def complete_authorization(
        self,
        connector_id: str,
        authorization_code: str
    ) -> dict:
        """Complete OAuth flow with authorization code."""
        
        if connector_id not in self.pending_authorizations:
            return {"error": "Unknown connector"}
        
        # Exchange code for token
        connector = self.client.connectors.update(
            connector_id,
            oauth={
                "code": authorization_code
            }
        )
        
        self.pending_authorizations[connector_id]["status"] = "authorized"
        
        return {
            "connector_id": connector_id,
            "status": "authorized",
            "type": connector.type
        }
    
    def get_status(self, connector_id: str) -> dict:
        """Get authorization status."""
        
        if connector_id in self.pending_authorizations:
            return self.pending_authorizations[connector_id]
        
        # Check with API
        try:
            connector = self.client.connectors.retrieve(connector_id)
            return {
                "connector_id": connector_id,
                "type": connector.type,
                "status": connector.status
            }
        except Exception as e:
            return {"error": str(e)}


# Usage
oauth_manager = OAuthFlowManager()

# Start authorization
# auth = oauth_manager.initiate_authorization(
#     ConnectorType.GOOGLE_CALENDAR,
#     redirect_uri="https://myapp.com/oauth/callback",
#     user_id="user_123"
# )
# 
# print(f"Redirect to: {auth['authorization_url']}")

# After user authorizes and is redirected back with code
# result = oauth_manager.complete_authorization(
#     connector_id=auth['connector_id'],
#     authorization_code="code_from_redirect"
# )
```

### OAuth Configuration

```python
@dataclass
class OAuthConfig:
    """OAuth configuration for a connector."""
    
    redirect_uri: str
    client_id: Optional[str] = None  # For custom OAuth apps
    client_secret: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    access_type: str = "offline"  # For refresh tokens
    prompt: str = "consent"
    
    def to_dict(self) -> dict:
        """Convert to API format."""
        config = {
            "redirect_uri": self.redirect_uri
        }
        
        if self.client_id:
            config["client_id"] = self.client_id
            config["client_secret"] = self.client_secret
        
        if self.scopes:
            config["scopes"] = self.scopes
        
        config["access_type"] = self.access_type
        config["prompt"] = self.prompt
        
        return config


class ConnectorFactory:
    """Factory for creating connectors with OAuth."""
    
    def __init__(self, default_redirect_uri: str):
        self.client = OpenAI()
        self.redirect_uri = default_redirect_uri
        self.connectors: Dict[str, dict] = {}
    
    def create_google_calendar(
        self,
        user_id: str,
        read_only: bool = False
    ) -> dict:
        """Create Google Calendar connector."""
        
        scopes = ["https://www.googleapis.com/auth/calendar.readonly"]
        if not read_only:
            scopes.append("https://www.googleapis.com/auth/calendar.events")
        
        return self._create_connector(
            ConnectorType.GOOGLE_CALENDAR,
            user_id,
            scopes
        )
    
    def create_gmail(
        self,
        user_id: str,
        send_enabled: bool = False
    ) -> dict:
        """Create Gmail connector."""
        
        scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
        if send_enabled:
            scopes.append("https://www.googleapis.com/auth/gmail.send")
        
        return self._create_connector(
            ConnectorType.GMAIL,
            user_id,
            scopes
        )
    
    def create_dropbox(self, user_id: str) -> dict:
        """Create Dropbox connector."""
        
        return self._create_connector(
            ConnectorType.DROPBOX,
            user_id,
            ["files.metadata.read", "files.content.read"]
        )
    
    def _create_connector(
        self,
        connector_type: ConnectorType,
        user_id: str,
        scopes: List[str]
    ) -> dict:
        """Create connector with specified config."""
        
        config = OAuthConfig(
            redirect_uri=self.redirect_uri,
            scopes=scopes
        )
        
        connector = self.client.connectors.create(
            type=connector_type.value,
            oauth=config.to_dict()
        )
        
        self.connectors[connector.id] = {
            "connector_id": connector.id,
            "type": connector_type.value,
            "user_id": user_id,
            "authorization_url": connector.authorization_url,
            "scopes": scopes
        }
        
        return self.connectors[connector.id]


# Usage
factory = ConnectorFactory("https://myapp.com/oauth/callback")

# Create calendar connector
# calendar = factory.create_google_calendar("user_123", read_only=False)
# print(f"Auth URL: {calendar['authorization_url']}")
```

---

## Using Connectors in Requests

### Basic Connector Usage

```python
from openai import OpenAI

client = OpenAI()

# Use connector in a request
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "connector",
        "connector_id": "conn_abc123"
    }],
    input="What meetings do I have tomorrow?"
)

print(response.output_text)
```

### Connector Request Manager

```python
@dataclass
class ConnectorRequest:
    """A request using connectors."""
    
    connectors: List[str]  # connector_ids
    query: str
    require_approval: str = "never"


class ConnectorRequestManager:
    """Manage requests that use connectors."""
    
    def __init__(self):
        self.client = OpenAI()
        self.active_connectors: Dict[str, dict] = {}
    
    def register_connector(
        self,
        connector_id: str,
        connector_type: str,
        user_id: str
    ):
        """Register an authorized connector."""
        
        self.active_connectors[connector_id] = {
            "id": connector_id,
            "type": connector_type,
            "user_id": user_id,
            "registered_at": datetime.now()
        }
    
    def query(
        self,
        user_id: str,
        query: str,
        connector_types: List[ConnectorType] = None
    ) -> dict:
        """Query using user's connectors."""
        
        # Get user's connectors
        user_connectors = [
            c for c in self.active_connectors.values()
            if c["user_id"] == user_id
        ]
        
        # Filter by type if specified
        if connector_types:
            type_values = [t.value for t in connector_types]
            user_connectors = [
                c for c in user_connectors
                if c["type"] in type_values
            ]
        
        if not user_connectors:
            return {"error": "No connectors available for user"}
        
        # Build tools config
        tools = [
            {"type": "connector", "connector_id": c["id"]}
            for c in user_connectors
        ]
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=tools,
            input=query
        )
        
        return {
            "output": response.output_text,
            "connectors_used": len(user_connectors),
            "connector_types": [c["type"] for c in user_connectors]
        }
    
    def query_calendar(self, user_id: str, query: str) -> dict:
        """Query calendar specifically."""
        return self.query(user_id, query, [ConnectorType.GOOGLE_CALENDAR])
    
    def query_email(self, user_id: str, query: str) -> dict:
        """Query email specifically."""
        return self.query(user_id, query, [ConnectorType.GMAIL, ConnectorType.MICROSOFT_OUTLOOK])
    
    def query_files(self, user_id: str, query: str) -> dict:
        """Query files specifically."""
        return self.query(
            user_id, 
            query, 
            [ConnectorType.GOOGLE_DRIVE, ConnectorType.DROPBOX, ConnectorType.MICROSOFT_SHAREPOINT]
        )


# Usage
manager = ConnectorRequestManager()

# Register connectors after OAuth
# manager.register_connector("conn_cal123", "google_calendar", "user_123")
# manager.register_connector("conn_mail456", "gmail", "user_123")

# Query
# result = manager.query_calendar("user_123", "What's on my calendar today?")
# print(result["output"])
```

---

## Managing Connector Lifecycles

### Connector Lifecycle Manager

```python
from datetime import datetime, timedelta

class ConnectorStatus(Enum):
    PENDING = "pending"
    AUTHORIZED = "authorized"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ERROR = "error"


@dataclass
class ManagedConnector:
    """A managed connector with lifecycle tracking."""
    
    connector_id: str
    connector_type: ConnectorType
    user_id: str
    status: ConnectorStatus
    created_at: datetime
    authorized_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    use_count: int = 0


class ConnectorLifecycleManager:
    """Manage connector lifecycles."""
    
    def __init__(self, token_refresh_before_expiry: timedelta = timedelta(hours=1)):
        self.client = OpenAI()
        self.connectors: Dict[str, ManagedConnector] = {}
        self.refresh_buffer = token_refresh_before_expiry
    
    def create(
        self,
        connector_type: ConnectorType,
        user_id: str,
        redirect_uri: str
    ) -> ManagedConnector:
        """Create a new connector."""
        
        connector = self.client.connectors.create(
            type=connector_type.value,
            oauth={"redirect_uri": redirect_uri}
        )
        
        managed = ManagedConnector(
            connector_id=connector.id,
            connector_type=connector_type,
            user_id=user_id,
            status=ConnectorStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.connectors[connector.id] = managed
        return managed
    
    def authorize(
        self,
        connector_id: str,
        authorization_code: str,
        token_lifetime: timedelta = timedelta(hours=1)
    ) -> bool:
        """Complete authorization."""
        
        if connector_id not in self.connectors:
            return False
        
        try:
            self.client.connectors.update(
                connector_id,
                oauth={"code": authorization_code}
            )
            
            managed = self.connectors[connector_id]
            managed.status = ConnectorStatus.AUTHORIZED
            managed.authorized_at = datetime.now()
            managed.expires_at = datetime.now() + token_lifetime
            
            return True
        except Exception:
            self.connectors[connector_id].status = ConnectorStatus.ERROR
            return False
    
    def refresh_if_needed(self, connector_id: str) -> bool:
        """Refresh token if close to expiry."""
        
        if connector_id not in self.connectors:
            return False
        
        managed = self.connectors[connector_id]
        
        if managed.status != ConnectorStatus.AUTHORIZED:
            return False
        
        if managed.expires_at is None:
            return True
        
        # Check if refresh needed
        if datetime.now() + self.refresh_buffer >= managed.expires_at:
            try:
                self.client.connectors.refresh(connector_id)
                managed.expires_at = datetime.now() + timedelta(hours=1)
                return True
            except Exception:
                managed.status = ConnectorStatus.EXPIRED
                return False
        
        return True
    
    def record_use(self, connector_id: str):
        """Record connector usage."""
        
        if connector_id in self.connectors:
            managed = self.connectors[connector_id]
            managed.last_used = datetime.now()
            managed.use_count += 1
    
    def revoke(self, connector_id: str) -> bool:
        """Revoke a connector."""
        
        if connector_id not in self.connectors:
            return False
        
        try:
            self.client.connectors.delete(connector_id)
            self.connectors[connector_id].status = ConnectorStatus.REVOKED
            return True
        except Exception:
            return False
    
    def get_user_connectors(self, user_id: str) -> List[ManagedConnector]:
        """Get all connectors for a user."""
        return [
            c for c in self.connectors.values()
            if c.user_id == user_id
        ]
    
    def get_active_connectors(self, user_id: str) -> List[ManagedConnector]:
        """Get active connectors for a user."""
        return [
            c for c in self.connectors.values()
            if c.user_id == user_id and c.status == ConnectorStatus.AUTHORIZED
        ]
    
    def cleanup_expired(self) -> int:
        """Clean up expired connectors."""
        
        count = 0
        now = datetime.now()
        
        for connector_id, managed in list(self.connectors.items()):
            if managed.expires_at and now > managed.expires_at:
                managed.status = ConnectorStatus.EXPIRED
                count += 1
        
        return count
    
    def get_statistics(self) -> dict:
        """Get connector statistics."""
        
        by_status = {}
        by_type = {}
        total_uses = 0
        
        for managed in self.connectors.values():
            # By status
            status = managed.status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            # By type
            ctype = managed.connector_type.value
            by_type[ctype] = by_type.get(ctype, 0) + 1
            
            total_uses += managed.use_count
        
        return {
            "total_connectors": len(self.connectors),
            "by_status": by_status,
            "by_type": by_type,
            "total_uses": total_uses
        }


# Usage
lifecycle = ConnectorLifecycleManager()

# Create connector
# managed = lifecycle.create(
#     ConnectorType.GOOGLE_CALENDAR,
#     "user_123",
#     "https://myapp.com/oauth/callback"
# )

# After OAuth callback
# lifecycle.authorize(managed.connector_id, "auth_code")

# Before using
# lifecycle.refresh_if_needed(managed.connector_id)

# Track usage
# lifecycle.record_use(managed.connector_id)

# Cleanup
# expired = lifecycle.cleanup_expired()
```

---

## Multi-Service Integration

### Combined Connector Client

```python
class MultiServiceConnectorClient:
    """Client that combines multiple connectors."""
    
    def __init__(self):
        self.client = OpenAI()
        self.lifecycle = ConnectorLifecycleManager()
        self.user_services: Dict[str, Dict[str, str]] = {}  # user_id -> {service: connector_id}
    
    def setup_user(
        self,
        user_id: str,
        redirect_uri: str,
        services: List[ConnectorType]
    ) -> Dict[str, str]:
        """Setup connectors for a user."""
        
        auth_urls = {}
        
        for service in services:
            managed = self.lifecycle.create(service, user_id, redirect_uri)
            
            if user_id not in self.user_services:
                self.user_services[user_id] = {}
            
            self.user_services[user_id][service.value] = managed.connector_id
            
            # Get auth URL
            connector = self.client.connectors.retrieve(managed.connector_id)
            auth_urls[service.value] = connector.authorization_url
        
        return auth_urls
    
    def get_user_tools(self, user_id: str) -> List[dict]:
        """Get tool configs for user's connectors."""
        
        active = self.lifecycle.get_active_connectors(user_id)
        
        tools = []
        for managed in active:
            # Refresh if needed
            self.lifecycle.refresh_if_needed(managed.connector_id)
            
            if managed.status == ConnectorStatus.AUTHORIZED:
                tools.append({
                    "type": "connector",
                    "connector_id": managed.connector_id
                })
        
        return tools
    
    def query_all_services(
        self,
        user_id: str,
        query: str
    ) -> dict:
        """Query across all user's services."""
        
        tools = self.get_user_tools(user_id)
        
        if not tools:
            return {"error": "No active connectors"}
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=tools,
            input=query
        )
        
        # Track usage
        for tool in tools:
            self.lifecycle.record_use(tool["connector_id"])
        
        return {
            "output": response.output_text,
            "services_used": len(tools)
        }
    
    def schedule_meeting(
        self,
        user_id: str,
        details: str
    ) -> dict:
        """Schedule a meeting using calendar connector."""
        
        calendar_connectors = [
            c for c in self.lifecycle.get_active_connectors(user_id)
            if c.connector_type in [
                ConnectorType.GOOGLE_CALENDAR,
                ConnectorType.MICROSOFT_OUTLOOK
            ]
        ]
        
        if not calendar_connectors:
            return {"error": "No calendar connector available"}
        
        tools = [
            {"type": "connector", "connector_id": c.connector_id}
            for c in calendar_connectors
        ]
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=tools,
            input=f"Schedule a meeting: {details}"
        )
        
        return {"output": response.output_text}
    
    def search_files(
        self,
        user_id: str,
        search_query: str
    ) -> dict:
        """Search files across all file services."""
        
        file_connectors = [
            c for c in self.lifecycle.get_active_connectors(user_id)
            if c.connector_type in [
                ConnectorType.GOOGLE_DRIVE,
                ConnectorType.DROPBOX,
                ConnectorType.MICROSOFT_SHAREPOINT
            ]
        ]
        
        if not file_connectors:
            return {"error": "No file connector available"}
        
        tools = [
            {"type": "connector", "connector_id": c.connector_id}
            for c in file_connectors
        ]
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=tools,
            input=f"Search for files: {search_query}"
        )
        
        return {"output": response.output_text}


# Usage
multi_client = MultiServiceConnectorClient()

# Setup user with multiple services
# auth_urls = multi_client.setup_user(
#     "user_123",
#     "https://myapp.com/oauth/callback",
#     [
#         ConnectorType.GOOGLE_CALENDAR,
#         ConnectorType.GMAIL,
#         ConnectorType.GOOGLE_DRIVE
#     ]
# )
# 
# print("Authorize at:")
# for service, url in auth_urls.items():
#     print(f"  {service}: {url}")

# After authorization, query all services
# result = multi_client.query_all_services(
#     "user_123",
#     "What meetings do I have today and any unread emails about the project?"
# )
```

---

## Hands-on Exercise

### Your Task

Build a connector management system for a productivity app.

### Requirements

1. Support multiple connector types
2. Handle OAuth flow
3. Manage connector lifecycle
4. Track usage and refresh tokens

<details>
<summary>💡 Hints</summary>

- Store connector state persistently
- Handle token refresh proactively
- Track which services each user has connected
</details>

<details>
<summary>✅ Solution</summary>

```python
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import json

class ServiceCategory(Enum):
    CALENDAR = "calendar"
    EMAIL = "email"
    FILES = "files"
    COMMUNICATION = "communication"


@dataclass
class UserConnectorProfile:
    """User's connector profile."""
    
    user_id: str
    connected_services: Dict[str, str] = field(default_factory=dict)  # type -> connector_id
    pending_authorizations: Dict[str, str] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class ProductivityConnectorSystem:
    """Complete connector system for productivity app."""
    
    def __init__(self, redirect_base_uri: str):
        self.client = OpenAI()
        self.redirect_base = redirect_base_uri
        
        self.users: Dict[str, UserConnectorProfile] = {}
        self.connectors: Dict[str, ManagedConnector] = {}
        
        # Service to category mapping
        self.service_categories = {
            ConnectorType.GOOGLE_CALENDAR: ServiceCategory.CALENDAR,
            ConnectorType.MICROSOFT_OUTLOOK: ServiceCategory.CALENDAR,
            ConnectorType.GMAIL: ServiceCategory.EMAIL,
            ConnectorType.GOOGLE_DRIVE: ServiceCategory.FILES,
            ConnectorType.DROPBOX: ServiceCategory.FILES,
            ConnectorType.MICROSOFT_SHAREPOINT: ServiceCategory.FILES,
            ConnectorType.MICROSOFT_TEAMS: ServiceCategory.COMMUNICATION
        }
    
    def get_or_create_user(self, user_id: str) -> UserConnectorProfile:
        """Get or create user profile."""
        
        if user_id not in self.users:
            self.users[user_id] = UserConnectorProfile(user_id=user_id)
        
        return self.users[user_id]
    
    def initiate_connection(
        self,
        user_id: str,
        service_type: ConnectorType
    ) -> dict:
        """Start connecting a service."""
        
        user = self.get_or_create_user(user_id)
        
        # Check if already connected
        if service_type.value in user.connected_services:
            return {
                "status": "already_connected",
                "connector_id": user.connected_services[service_type.value]
            }
        
        # Create connector
        redirect_uri = f"{self.redirect_base}/oauth/callback/{user_id}"
        
        connector = self.client.connectors.create(
            type=service_type.value,
            oauth={"redirect_uri": redirect_uri}
        )
        
        # Track pending
        user.pending_authorizations[service_type.value] = connector.id
        
        self.connectors[connector.id] = ManagedConnector(
            connector_id=connector.id,
            connector_type=service_type,
            user_id=user_id,
            status=ConnectorStatus.PENDING,
            created_at=datetime.now()
        )
        
        return {
            "status": "pending",
            "connector_id": connector.id,
            "authorization_url": connector.authorization_url,
            "service": service_type.value
        }
    
    def complete_connection(
        self,
        user_id: str,
        service_type: ConnectorType,
        authorization_code: str
    ) -> dict:
        """Complete OAuth flow."""
        
        user = self.get_or_create_user(user_id)
        
        if service_type.value not in user.pending_authorizations:
            return {"error": "No pending authorization for this service"}
        
        connector_id = user.pending_authorizations[service_type.value]
        
        try:
            self.client.connectors.update(
                connector_id,
                oauth={"code": authorization_code}
            )
            
            # Update tracking
            user.connected_services[service_type.value] = connector_id
            del user.pending_authorizations[service_type.value]
            
            managed = self.connectors[connector_id]
            managed.status = ConnectorStatus.AUTHORIZED
            managed.authorized_at = datetime.now()
            managed.expires_at = datetime.now() + timedelta(hours=1)
            
            return {
                "status": "connected",
                "connector_id": connector_id,
                "service": service_type.value
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def disconnect_service(
        self,
        user_id: str,
        service_type: ConnectorType
    ) -> dict:
        """Disconnect a service."""
        
        user = self.get_or_create_user(user_id)
        
        if service_type.value not in user.connected_services:
            return {"error": "Service not connected"}
        
        connector_id = user.connected_services[service_type.value]
        
        try:
            self.client.connectors.delete(connector_id)
            
            del user.connected_services[service_type.value]
            
            if connector_id in self.connectors:
                self.connectors[connector_id].status = ConnectorStatus.REVOKED
            
            return {"status": "disconnected", "service": service_type.value}
        
        except Exception as e:
            return {"error": str(e)}
    
    def get_user_services(self, user_id: str) -> dict:
        """Get user's connected services by category."""
        
        user = self.get_or_create_user(user_id)
        
        by_category = {cat.value: [] for cat in ServiceCategory}
        
        for service_type, connector_id in user.connected_services.items():
            try:
                conn_type = ConnectorType(service_type)
                category = self.service_categories.get(conn_type)
                
                if category:
                    by_category[category.value].append({
                        "service": service_type,
                        "connector_id": connector_id
                    })
            except ValueError:
                pass
        
        return by_category
    
    def _ensure_tokens_fresh(self, user_id: str):
        """Refresh tokens if needed."""
        
        user = self.get_or_create_user(user_id)
        
        for connector_id in user.connected_services.values():
            if connector_id in self.connectors:
                managed = self.connectors[connector_id]
                
                if managed.expires_at and datetime.now() + timedelta(minutes=10) >= managed.expires_at:
                    try:
                        self.client.connectors.refresh(connector_id)
                        managed.expires_at = datetime.now() + timedelta(hours=1)
                    except Exception:
                        managed.status = ConnectorStatus.EXPIRED
    
    def query(
        self,
        user_id: str,
        query: str,
        categories: List[ServiceCategory] = None
    ) -> dict:
        """Query user's connected services."""
        
        self._ensure_tokens_fresh(user_id)
        
        user = self.get_or_create_user(user_id)
        
        # Get relevant connectors
        tools = []
        services_used = []
        
        for service_type, connector_id in user.connected_services.items():
            try:
                conn_type = ConnectorType(service_type)
                category = self.service_categories.get(conn_type)
                
                # Filter by category if specified
                if categories and category not in categories:
                    continue
                
                # Check status
                managed = self.connectors.get(connector_id)
                if managed and managed.status == ConnectorStatus.AUTHORIZED:
                    tools.append({
                        "type": "connector",
                        "connector_id": connector_id
                    })
                    services_used.append(service_type)
            except ValueError:
                pass
        
        if not tools:
            return {"error": "No active connectors for query"}
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=tools,
            input=query
        )
        
        # Track usage
        for connector_id in [t["connector_id"] for t in tools]:
            if connector_id in self.connectors:
                self.connectors[connector_id].last_used = datetime.now()
                self.connectors[connector_id].use_count += 1
        
        return {
            "output": response.output_text,
            "services_used": services_used
        }
    
    def get_calendar_summary(self, user_id: str) -> dict:
        """Get calendar summary."""
        return self.query(
            user_id,
            "Summarize my calendar for today and tomorrow",
            [ServiceCategory.CALENDAR]
        )
    
    def get_unread_emails(self, user_id: str) -> dict:
        """Get unread email summary."""
        return self.query(
            user_id,
            "What are my important unread emails?",
            [ServiceCategory.EMAIL]
        )
    
    def search_files(self, user_id: str, search_term: str) -> dict:
        """Search across file services."""
        return self.query(
            user_id,
            f"Find files related to: {search_term}",
            [ServiceCategory.FILES]
        )
    
    def daily_briefing(self, user_id: str) -> dict:
        """Get daily productivity briefing."""
        return self.query(
            user_id,
            "Give me a daily briefing: today's calendar, important emails, and any urgent items"
        )
    
    def get_system_stats(self) -> dict:
        """Get system-wide statistics."""
        
        total_users = len(self.users)
        total_connectors = len(self.connectors)
        
        by_status = {}
        by_service = {}
        total_uses = 0
        
        for managed in self.connectors.values():
            status = managed.status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            service = managed.connector_type.value
            by_service[service] = by_service.get(service, 0) + 1
            
            total_uses += managed.use_count
        
        return {
            "total_users": total_users,
            "total_connectors": total_connectors,
            "by_status": by_status,
            "by_service": by_service,
            "total_api_calls": total_uses
        }
    
    def export_user_data(self, user_id: str) -> str:
        """Export user's connector data."""
        
        user = self.get_or_create_user(user_id)
        
        data = {
            "user_id": user_id,
            "connected_services": list(user.connected_services.keys()),
            "pending_authorizations": list(user.pending_authorizations.keys()),
            "created_at": user.created_at.isoformat(),
            "connectors": [
                {
                    "service": c.connector_type.value,
                    "status": c.status.value,
                    "use_count": c.use_count,
                    "last_used": c.last_used.isoformat() if c.last_used else None
                }
                for cid, c in self.connectors.items()
                if c.user_id == user_id
            ]
        }
        
        return json.dumps(data, indent=2)


# Usage example
system = ProductivityConnectorSystem("https://myapp.com")

# Setup new user
user_id = "user_123"

# Connect Google services
# auth1 = system.initiate_connection(user_id, ConnectorType.GOOGLE_CALENDAR)
# auth2 = system.initiate_connection(user_id, ConnectorType.GMAIL)
# auth3 = system.initiate_connection(user_id, ConnectorType.GOOGLE_DRIVE)

# print("Authorization URLs:")
# print(f"Calendar: {auth1['authorization_url']}")
# print(f"Gmail: {auth2['authorization_url']}")
# print(f"Drive: {auth3['authorization_url']}")

# After OAuth callbacks
# system.complete_connection(user_id, ConnectorType.GOOGLE_CALENDAR, "code1")
# system.complete_connection(user_id, ConnectorType.GMAIL, "code2")
# system.complete_connection(user_id, ConnectorType.GOOGLE_DRIVE, "code3")

# Get user's services
services = system.get_user_services(user_id)
print(f"Connected services: {services}")

# Daily briefing
# briefing = system.daily_briefing(user_id)
# print(briefing['output'])

# System stats
stats = system.get_system_stats()
print(f"Total users: {stats['total_users']}")
print(f"Total connectors: {stats['total_connectors']}")
```

</details>

---

## Summary

✅ OpenAI Connectors integrate with Google, Microsoft, and Dropbox  
✅ OAuth handles secure authorization  
✅ connector_id references authorized connections  
✅ Lifecycle management tracks status and refresh  
✅ Multiple connectors can be combined in requests  
✅ Token refresh keeps connections active

**Next:** [MCP Security](./08-mcp-security.md)

---

## Further Reading

- [OpenAI Connectors Guide](https://platform.openai.com/docs/guides/connectors) — Official documentation
- [OAuth 2.0 Overview](https://oauth.net/2/) — OAuth specification
- [Google OAuth Scopes](https://developers.google.com/identity/protocols/oauth2/scopes) — Scope reference
