---
title: "API Key Generation and Management"
---

# API Key Generation and Management

## Introduction

API keys are the gateway to AI services. Before making your first API call, you need to generate a key from your provider's dashboard. This lesson walks through the key generation process for major providers and explains how to organize keys for different use cases.

Getting this right from the start saves headaches later—properly named and scoped keys make debugging easier and security incidents more manageable.

### What We'll Cover

- Creating API keys in OpenAI, Anthropic, and Google dashboards
- Naming conventions for key organization
- Using multiple keys for different environments
- Understanding key permissions and scopes
- Revoking compromised keys safely
- Project-scoped keys (OpenAI) and workspaces (Anthropic)

### Prerequisites

- Accounts with at least one AI provider (OpenAI, Anthropic, or Google)
- Credit card on file (most providers require payment method)
- Understanding of development vs. production environments

---

## OpenAI API Key Generation

OpenAI manages API keys at the organization and project level through their [platform dashboard](https://platform.openai.com/).

### Step-by-Step: Creating an OpenAI Key

1. Navigate to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click **"Create new secret key"**
3. Configure the key:
   - **Name**: Descriptive identifier (e.g., `dev-laptop-john`, `prod-backend-v2`)
   - **Project**: Select or create a project for scoping
   - **Permissions**: Choose between "All" or restricted permissions

```mermaid
flowchart LR
    A[Dashboard] --> B[API Keys]
    B --> C[Create new secret key]
    C --> D{Configure}
    D --> E[Name]
    D --> F[Project]
    D --> G[Permissions]
    E & F & G --> H[Generate]
    H --> I[Copy & Store Securely]
```

> **Important:** OpenAI shows your key only once. Copy it immediately and store it securely. If you lose it, you'll need to generate a new one.

### OpenAI Key Format

OpenAI keys follow a recognizable pattern:

| Key Type | Pattern | Example |
|----------|---------|---------|
| Project key | `sk-proj-...` | `sk-proj-abc123def456...` |
| User key (legacy) | `sk-...` | `sk-abc123def456...` |
| Service account | `sk-svcacct-...` | `sk-svcacct-org-abc...` |

> **Note:** Project-scoped keys (`sk-proj-`) are preferred for new applications. They provide better isolation and more granular billing.

---

## Anthropic API Key Generation

Anthropic uses the [Console](https://console.anthropic.com/) for key management, with workspaces for organization.

### Step-by-Step: Creating an Anthropic Key

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Navigate to **Settings** → **API Keys**
3. Click **"Create Key"**
4. Enter a descriptive name
5. Select the workspace (for organization accounts)

```python
# Anthropic keys start with 'sk-ant-'
# Example format:
# sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import anthropic

# The SDK reads ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic()
```

### Workspaces for Team Organization

Anthropic workspaces let you:
- Segment API keys by team or project
- Set separate spend limits per workspace
- Track usage independently

| Workspace | Purpose | Spend Limit |
|-----------|---------|-------------|
| `development` | Testing and prototyping | $100/month |
| `production` | Live application traffic | $5,000/month |
| `research` | Experimentation | $500/month |

---

## Google Gemini API Key Generation

Google provides API keys through [AI Studio](https://aistudio.google.com/) or Google Cloud Console.

### Option 1: AI Studio (Simpler)

1. Visit [aistudio.google.com](https://aistudio.google.com/)
2. Click **"Get API Key"** in the left sidebar
3. Click **"Create API key"**
4. Select or create a Google Cloud project

### Option 2: Google Cloud Console (More Control)

1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. Navigate to **APIs & Services** → **Credentials**
3. Click **"Create credentials"** → **"API Key"**
4. Restrict the key to Generative AI API only

```javascript
// Google AI SDK usage
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY });
```

> **Tip:** Google Cloud keys can be restricted to specific APIs, IP addresses, and referrer URLs. Use these restrictions in production.

---

## Key Naming Conventions

Good naming makes key management easier. Use a consistent pattern across all providers:

```
{environment}-{purpose}-{identifier}
```

| Key Name | Environment | Purpose | Identifier |
|----------|-------------|---------|------------|
| `prod-backend-api-v2` | Production | Backend service | Version 2 |
| `dev-local-john` | Development | Local development | Developer name |
| `staging-ci-github` | Staging | CI/CD pipeline | Platform |
| `test-integration-suite` | Test | Integration tests | Test suite |

### Anti-Patterns to Avoid

| ❌ Bad Name | ✅ Better Name | Why |
|-------------|----------------|-----|
| `key1` | `dev-local-laptop` | Descriptive |
| `test` | `test-integration-payment` | Specific purpose |
| `johns-key` | `dev-local-john-2024` | Includes context |
| `API_KEY` | `prod-backend-orders` | Identifies usage |

---

## Managing Multiple Keys

Most applications need multiple keys for different environments:

```mermaid
flowchart TD
    subgraph Development
        A[dev-local-laptop]
        B[dev-local-desktop]
    end
    
    subgraph CI/CD
        C[staging-ci-github]
        D[staging-ci-jenkins]
    end
    
    subgraph Production
        E[prod-backend-primary]
        F[prod-backend-backup]
    end
    
    A & B --> |"Lower limits"| DevEnv[Development Environment]
    C & D --> |"Medium limits"| StagingEnv[Staging Environment]
    E & F --> |"Full limits"| ProdEnv[Production Environment]
```

### Benefits of Environment Separation

1. **Cost isolation**: Dev mistakes don't affect production billing
2. **Rate limit separation**: Testing doesn't consume production limits
3. **Security containment**: Compromised dev key can't access prod data
4. **Audit clarity**: Know exactly which system made each request

---

## Key Permissions and Scopes

Modern AI platforms offer granular permissions. Use the principle of least privilege.

### OpenAI Permission Levels

| Permission | Access Level | Use Case |
|------------|--------------|----------|
| All | Full API access | Admin, full integration |
| Read | Read-only operations | Monitoring, analytics |
| Restricted | Specific endpoints only | Single-purpose services |

### Example: Restricted Key for Embeddings Only

```javascript
// This key only has access to embeddings endpoint
const embeddingsClient = new OpenAI({
  apiKey: process.env.OPENAI_EMBEDDINGS_KEY, // Restricted scope
});

// This would fail with a restricted key:
// embeddingsClient.chat.completions.create(...) // 403 Forbidden
```

---

## Revoking Compromised Keys

If a key is exposed (committed to GitHub, shared accidentally, etc.), revoke it immediately.

### Immediate Response Steps

1. **Revoke the key** in the provider dashboard
2. **Generate a new key** with the same permissions
3. **Update all systems** using the old key
4. **Audit usage** for unauthorized activity
5. **Check billing** for unexpected charges

```mermaid
flowchart LR
    A[Key Exposed] --> B[Revoke Immediately]
    B --> C[Generate New Key]
    C --> D[Update Systems]
    D --> E[Audit Usage]
    E --> F[Check Billing]
```

### OpenAI Key Revocation

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Find the compromised key
3. Click the trash icon or "Revoke"
4. Confirm revocation

> **Warning:** Revocation is immediate. Any systems using the key will fail instantly. Have a replacement ready before revoking production keys.

---

## Project-Scoped Keys (OpenAI)

OpenAI's project system provides additional organization:

```mermaid
flowchart TD
    Org[Organization] --> P1[Project: ChatBot]
    Org --> P2[Project: Analytics]
    Org --> P3[Project: Internal Tools]
    
    P1 --> K1[Key: chatbot-prod]
    P1 --> K2[Key: chatbot-dev]
    
    P2 --> K3[Key: analytics-backend]
    
    P3 --> K4[Key: tools-slack-bot]
```

### Creating a Project

1. Visit [platform.openai.com/settings](https://platform.openai.com/settings)
2. Click **"Create project"**
3. Name the project (e.g., "Customer Support Bot")
4. Generate keys scoped to this project

### Benefits of Project Scoping

| Feature | Benefit |
|---------|---------|
| Isolated billing | Track costs per project |
| Separate rate limits | Projects don't compete for quota |
| Team access control | Different teams, different projects |
| Usage attribution | Know which project consumed tokens |

---

## Service Account Authentication (Google Cloud)

For production Google Cloud deployments, service accounts are more secure than API keys:

```python
# Using service account (recommended for GCP)
from google.cloud import aiplatform

# Authenticates using GOOGLE_APPLICATION_CREDENTIALS env var
aiplatform.init(project="my-project", location="us-central1")
```

### Service Account vs. API Key

| Aspect | API Key | Service Account |
|--------|---------|-----------------|
| Setup complexity | Simple | More complex |
| Security | Basic | Fine-grained IAM |
| Rotation | Manual | Automatic possible |
| Audit logging | Limited | Comprehensive |
| Best for | Development, simple apps | Production GCP |

---

## Hands-on Exercise

### Your Task

Create a key management strategy for a new AI-powered application:

1. Sign up for OpenAI or Anthropic (if you haven't)
2. Create three keys with proper naming:
   - `dev-local-{yourname}`
   - `test-integration-{appname}`
   - `staging-preview-{appname}`
3. Document each key's purpose and scope
4. Practice revoking one key and replacing it

### Requirements

- Each key has a descriptive name following the convention
- Keys are stored in a password manager (not plain text)
- At least one key has restricted permissions (if available)

### Expected Result

You should have:
- 3 properly named API keys
- A document mapping keys to their purposes
- Experience with the revocation/replacement process

<details>
<summary>💡 Hints</summary>

- Use a password manager like 1Password, Bitwarden, or the built-in OS keychain
- Check the API key settings for permission/scope options
- Export keys as environment variables for testing:
  ```bash
  export OPENAI_API_KEY="sk-proj-your-dev-key..."
  ```
</details>

<details>
<summary>✅ Example Key Documentation</summary>

| Key Name | Provider | Purpose | Permissions | Created | Notes |
|----------|----------|---------|-------------|---------|-------|
| dev-local-john | OpenAI | Local development | All | 2025-01-15 | John's laptop |
| test-integration-chatbot | OpenAI | CI tests | Restricted | 2025-01-15 | GitHub Actions |
| staging-preview-chatbot | OpenAI | Preview deploys | All | 2025-01-15 | Vercel previews |

</details>

---

## Summary

✅ Generate API keys from each provider's dashboard with descriptive names  
✅ Use naming conventions like `{env}-{purpose}-{identifier}` for clarity  
✅ Create separate keys for development, staging, and production  
✅ Apply least-privilege permissions when available  
✅ Revoke compromised keys immediately and have replacement ready  
✅ Use project-scoped keys (OpenAI) or workspaces (Anthropic) for organization

**Next:** [Authentication Methods](./02-authentication-methods.md)

---

## Further Reading

- [OpenAI API Keys Dashboard](https://platform.openai.com/api-keys) — Manage OpenAI keys
- [Anthropic Console Settings](https://console.anthropic.com/settings/keys) — Anthropic key management
- [Google AI Studio](https://aistudio.google.com/) — Gemini API key generation
- [OpenAI Projects Guide](https://platform.openai.com/docs/guides/production-best-practices) — Project organization

<!-- 
Sources Consulted:
- OpenAI API Reference Authentication: https://platform.openai.com/docs/api-reference/authentication
- Anthropic API Getting Started: https://platform.claude.com/docs/en/api/getting-started
- Google AI Studio documentation
-->
