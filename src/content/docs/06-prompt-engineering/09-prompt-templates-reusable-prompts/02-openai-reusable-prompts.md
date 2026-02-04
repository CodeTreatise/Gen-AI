---
title: "OpenAI Reusable Prompts"
---

# OpenAI Reusable Prompts

## Introduction

OpenAI's Dashboard provides a no-code approach to prompt management. Create prompts with variables, manage versions, share with teams, and deploy changes without touching code. This is ideal for teams where prompt iteration happens faster than code deployments, or where non-engineers need to optimize prompts.

> **🔑 Key Insight:** Reusable prompts separate prompt development from application code. Your integration code stays stable while prompts evolve in the Dashboard.

### What We'll Cover

- Creating prompts in the OpenAI Dashboard
- Variable syntax and substitution
- Version management
- Team sharing and collaboration
- A/B testing prompt versions
- API integration patterns

### Prerequisites

- OpenAI API account with Dashboard access
- [Prompt Templates Overview](./00-prompt-templates-overview.md)

---

## Dashboard Overview

### Creating a Reusable Prompt

1. Navigate to [platform.openai.com/chat/edit](https://platform.openai.com/chat/edit)
2. Click "Create new prompt"
3. Define your prompt with placeholders
4. Save and get your prompt ID

```
┌─────────────────────────────────────────────────────────────┐
│  OpenAI Dashboard - Prompt Editor                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Prompt Name: [customer-support-response]                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ You are a helpful customer support agent for        │   │
│  │ {{company_name}}.                                   │   │
│  │                                                     │   │
│  │ Customer: {{customer_name}}                         │   │
│  │ Issue: {{issue_description}}                        │   │
│  │                                                     │   │
│  │ Respond helpfully and professionally.               │   │
│  │ If you cannot help, escalate to a human agent.      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Variables detected: company_name, customer_name,           │
│                      issue_description                      │
│                                                             │
│  [Test] [Save] [Deploy as Current]                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Variable Syntax

### Basic Variables

Use double curly braces for variable placeholders:

```
Hello {{customer_name}},

Thank you for purchasing {{product_name}}.

Your order number is: {{order_id}}
```

### Variable Types

Variables can accept different content types:

| Type | Example | Use Case |
|------|---------|----------|
| String | `{{customer_name}}` | Text values |
| Text block | `{{document_content}}` | Long-form content |
| Image | `{{product_image}}` | Vision tasks |
| File | `{{uploaded_document}}` | Document analysis |

### API Variable Substitution

```javascript
import OpenAI from "openai";
const client = new OpenAI();

// String variables
const response = await client.responses.create({
    model: "gpt-5",
    prompt: {
        id: "pmpt_abc123",
        version: "2",
        variables: {
            customer_name: "Jane Doe",
            product_name: "Premium Plan",
            order_id: "ORD-12345"
        }
    }
});

console.log(response.output_text);
```

### File/Image Variables

```javascript
// Variables with file input
const response = await client.responses.create({
    model: "gpt-5",
    prompt: {
        id: "pmpt_document_analyzer",
        variables: {
            analysis_type: "summary",
            document: {
                type: "input_file",
                file_id: "file-abc123"
            }
        }
    }
});
```

---

## Version Management

### How Versioning Works

```
Prompt: customer-support-response
├── Version 1 (created Jan 15)
│   └── Original prompt
├── Version 2 (created Jan 20) ← Current
│   └── Added escalation instructions
├── Version 3 (created Jan 25)
│   └── Improved tone guidelines
└── Version 4 (draft)
    └── Testing new format
```

### Setting the Current Version

In the Dashboard:
1. Open your prompt
2. View version history
3. Select a version
4. Click "Set as Current"

### Using Specific Versions in API

```javascript
// Use the "current" version (default)
const response = await client.responses.create({
    model: "gpt-5",
    prompt: {
        id: "pmpt_abc123"
        // version omitted = uses "current"
    }
});

// Pin to a specific version
const response = await client.responses.create({
    model: "gpt-5",
    prompt: {
        id: "pmpt_abc123",
        version: "2"  // Always use version 2
    }
});
```

### Version Best Practices

| Practice | Reason |
|----------|--------|
| Pin production to specific version | Avoid unexpected changes |
| Use "current" for development | Get latest improvements |
| Document version changes | Track what changed and why |
| Test before setting as current | Validate with real inputs |

---

## Team Sharing & Collaboration

### Access Control

```
Organization: Acme Corp
├── Team: AI Product
│   ├── Member: Alice (Admin) - Can edit prompts
│   ├── Member: Bob (Member) - Can use prompts
│   └── Member: Carol (Member) - Can use prompts
└── Team: Customer Success
    └── Member: Dave (Viewer) - Can view only
```

### Collaborative Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                 PROMPT DEVELOPMENT WORKFLOW                  │
└─────────────────────────────────────────────────────────────┘
                              │
         1. Draft             │          4. Deploy
    ┌──────────────┐          │     ┌──────────────┐
    │  Create new  │──────────┼────▶│ Set as       │
    │  version     │          │     │ current      │
    └──────────────┘          │     └──────────────┘
            │                 │            │
            ▼                 │            ▼
    ┌──────────────┐          │     ┌──────────────┐
    │  Test in     │──────────┼────▶│ Production   │
    │  Playground  │          │     │ traffic      │
    └──────────────┘          │     └──────────────┘
            │                 │
         2. Review            │
    ┌──────────────┐          │
    │  Share with  │──────────┘
    │  team        │
    └──────────────┘
            │
         3. Iterate
    ┌──────────────┐
    │  Refine      │
    │  based on    │
    │  feedback    │
    └──────────────┘
```

---

## A/B Testing

### Testing Different Versions

Run experiments to compare prompt performance:

```javascript
async function getPromptVersion(userId) {
    // Simple A/B split
    const hash = hashUserId(userId);
    return hash % 2 === 0 ? "3" : "4";  // 50/50 split
}

async function handleRequest(userId, query) {
    const version = await getPromptVersion(userId);
    
    const response = await client.responses.create({
        model: "gpt-5",
        prompt: {
            id: "pmpt_search_assistant",
            version: version,
            variables: { query }
        }
    });
    
    // Log for analysis
    await logExperiment({
        userId,
        promptVersion: version,
        query,
        response: response.output_text
    });
    
    return response.output_text;
}
```

### Weighted Distribution

```javascript
function selectVersion(weights) {
    // weights = { "3": 0.7, "4": 0.2, "5": 0.1 }
    const random = Math.random();
    let cumulative = 0;
    
    for (const [version, weight] of Object.entries(weights)) {
        cumulative += weight;
        if (random < cumulative) {
            return version;
        }
    }
    
    return Object.keys(weights)[0]; // Fallback
}

// 70% version 3, 20% version 4, 10% version 5
const version = selectVersion({
    "3": 0.7,
    "4": 0.2,
    "5": 0.1
});
```

### Measuring Results

Track metrics per version:

```javascript
const experimentMetrics = {
    "version_3": {
        requests: 0,
        avgLatency: 0,
        userRatings: [],
        errorRate: 0
    },
    "version_4": {
        requests: 0,
        avgLatency: 0,
        userRatings: [],
        errorRate: 0
    }
};

async function trackExperiment(version, latency, rating, error) {
    const metrics = experimentMetrics[`version_${version}`];
    metrics.requests++;
    metrics.avgLatency = (metrics.avgLatency * (metrics.requests - 1) + latency) / metrics.requests;
    if (rating) metrics.userRatings.push(rating);
    if (error) metrics.errorRate = (metrics.errorRate * (metrics.requests - 1) + 1) / metrics.requests;
}
```

---

## Integration Patterns

### Environment-Based Versioning

```javascript
const PROMPT_VERSIONS = {
    development: undefined,  // Use "current" 
    staging: "5",           // Latest stable
    production: "4"         // Proven version
};

async function callPrompt(promptId, variables) {
    const env = process.env.NODE_ENV || 'development';
    
    return client.responses.create({
        model: "gpt-5",
        prompt: {
            id: promptId,
            version: PROMPT_VERSIONS[env],
            variables
        }
    });
}
```

### Fallback Pattern

```javascript
async function callWithFallback(promptId, variables) {
    try {
        return await client.responses.create({
            model: "gpt-5",
            prompt: {
                id: promptId,
                variables
            }
        });
    } catch (error) {
        console.error("Prompt failed, using fallback:", error);
        
        // Fallback to inline prompt
        return await client.responses.create({
            model: "gpt-5",
            input: [
                {
                    role: "developer",
                    content: `Respond helpfully to: ${variables.query}`
                }
            ]
        });
    }
}
```

### Caching Integration

```javascript
async function callWithCaching(promptId, variables) {
    const response = await client.responses.create({
        model: "gpt-5",
        prompt: {
            id: promptId,
            variables
        }
    });
    
    // Check cache utilization
    const usage = response.usage;
    console.log({
        cached: usage.prompt_tokens_details.cached_tokens,
        total: usage.prompt_tokens,
        cacheRate: usage.prompt_tokens_details.cached_tokens / usage.prompt_tokens
    });
    
    return response;
}
```

---

## When to Use Dashboard Prompts

### Good Fit ✅

| Scenario | Why |
|----------|-----|
| Rapid iteration | Change prompts without deployments |
| Non-engineer collaboration | Content teams can edit prompts |
| A/B testing | Test versions without code changes |
| Prompt auditing | Built-in version history |
| Shared prompts | Same prompt across multiple applications |

### Consider Alternatives ❌

| Scenario | Better Approach |
|----------|-----------------|
| Complex conditional logic | Code-based templates |
| CI/CD integration needed | File-based with version control |
| Multi-provider deployment | Provider-agnostic library |
| Offline development | Local template files |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Name prompts descriptively | `customer-support-billing` not `prompt-1` |
| Document each version | Explain what changed and why |
| Test before deploying | Use Playground with real inputs |
| Pin production versions | Avoid surprise behavior changes |
| Review team access | Limit who can modify production prompts |
| Monitor usage | Track which versions are active |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Not pinning versions | Production breaks on update | Always specify version in production |
| Missing variables | Runtime errors | Validate all variables before calling |
| No rollback plan | Stuck with broken prompt | Keep previous version documented |
| Over-reliance on dashboard | Can't work offline | Have local backup templates |

---

## Hands-on Exercise

### Your Task

Create a reusable prompt in the OpenAI Dashboard for product descriptions that:
1. Takes `product_name`, `features` (list), and `target_audience`
2. Generates marketing-focused descriptions
3. Create two versions: formal and casual tone
4. Set up A/B testing between versions

<details>
<summary>💡 Hints</summary>

1. Create the prompt with all three variables
2. Version 1: formal corporate tone
3. Version 2: casual, friendly tone
4. Use user ID hashing for consistent assignment

</details>

<details>
<summary>✅ Solution</summary>

**Version 1 (Formal):**
```
You are a professional copywriter for enterprise B2B products.

Product: {{product_name}}
Key Features:
{{features}}

Target Audience: {{target_audience}}

Write a professional product description that:
- Uses formal business language
- Emphasizes ROI and efficiency
- Includes a clear call to action
- Is approximately 150 words
```

**Version 2 (Casual):**
```
You're a friendly copywriter who makes products sound exciting!

Product: {{product_name}}
Key Features:
{{features}}

Target Audience: {{target_audience}}

Write a fun, engaging product description that:
- Uses conversational language
- Highlights how it makes life easier
- Feels approachable and human
- Is approximately 150 words
```

**A/B Testing Code:**
```javascript
function getProductDescriptionVersion(userId) {
    // Consistent assignment per user
    const hash = userId.split('').reduce((a, b) => {
        a = ((a << 5) - a) + b.charCodeAt(0);
        return a & a;
    }, 0);
    
    return Math.abs(hash) % 2 === 0 ? "1" : "2";
}
```

</details>

---

## Summary

- OpenAI Dashboard enables no-code prompt management
- Variables use `{{variable_name}}` syntax
- Version management provides rollback and history
- A/B testing compares prompt performance
- Pin production to specific versions for stability
- Team sharing enables collaboration without code access

**Next:** [Prompt Libraries](./03-prompt-libraries.md)

---

<!-- Sources: OpenAI Prompt Engineering Guide (Reusable Prompts section), OpenAI API Reference -->
