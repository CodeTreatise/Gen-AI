---
title: "Role and Context Setting"
---

# Role and Context Setting

## Introduction

Setting a role tells the model *who* it should be. Providing context tells it *what situation* it's in. Together, these shape how the model approaches your task — its expertise level, communication style, and decision-making framework.

### What We'll Cover

- "You are a..." patterns
- Domain expertise assignment
- Audience definition
- Background information
- Context placement strategies

### Prerequisites

- [Clarity and Specificity](./01-clarity-and-specificity.md)

---

## Why Roles Matter

The same question gets different answers depending on the assigned role:

```
Question: "How should I invest $10,000?"

Role: Financial advisor
→ Diversified portfolio, risk assessment, time horizon questions

Role: Day trader  
→ Technical analysis, momentum stocks, short-term plays

Role: Frugal retiree
→ Low-risk bonds, dividend stocks, capital preservation
```

Roles activate different "knowledge frames" in the model.

---

## The "You Are a..." Pattern

### Basic Structure

```
You are a [role] with expertise in [domain]. 

Your goal is to [objective].

You communicate in a [style] manner.
```

### Examples

```markdown
# Senior Code Reviewer
You are a senior software engineer with 15 years of experience 
in Python and distributed systems.

Your goal is to identify bugs, security issues, and 
maintainability problems in code.

You communicate directly, prioritizing the most critical 
issues first. You explain *why* something is a problem, 
not just *that* it's a problem.
```

```markdown
# Medical Information Assistant
You are a medical information assistant that helps users 
understand health topics.

Your goal is to provide accurate, evidence-based information 
while always recommending professional consultation.

You communicate in clear, accessible language without 
unnecessary jargon. You never diagnose or prescribe.
```

---

## Domain Expertise Assignment

Specify the expertise level and specialization:

### Expertise Levels

| Level | Characteristics |
|-------|-----------------|
| **Beginner** | Simple explanations, avoids jargon, step-by-step |
| **Intermediate** | Can use technical terms, assumes basic knowledge |
| **Expert** | Advanced concepts, nuanced discussion, edge cases |
| **Specialist** | Deep domain knowledge, cutting-edge techniques |

### Example

```markdown
# Beginner-friendly
You are a Python tutor helping complete beginners. Explain 
every concept from first principles. Avoid jargon. Use 
analogies to everyday objects.

# Expert-level
You are a Python performance engineer. Assume familiarity 
with CPython internals, profiling tools, and optimization 
techniques. Focus on microsecond-level improvements.
```

### Domain Combinations

```markdown
You are a data scientist specializing in:
- Time series forecasting
- Python (pandas, scikit-learn, Prophet)
- Financial market data

You have particular expertise in handling:
- Missing data in high-frequency trading datasets
- Feature engineering for OHLCV data
- Backtesting methodologies
```

---

## Audience Definition

Tell the model who it's communicating with:

### Audience Factors

| Factor | Examples |
|--------|----------|
| **Knowledge level** | Beginner, intermediate, expert |
| **Role** | Developer, manager, end-user, student |
| **Goals** | Learning, problem-solving, decision-making |
| **Constraints** | Time-limited, mobile reader, accessibility needs |

### Examples

```markdown
# For developers
Explain this API to developers who will integrate it. 
Include code examples, error handling, and edge cases.

# For managers
Explain this API to engineering managers evaluating it. 
Focus on capabilities, limitations, costs, and maintenance burden.

# For end-users
Explain what this feature does to users of our mobile app. 
Avoid technical jargon. Focus on benefits and how-to steps.
```

### Adjusting for Audience

```markdown
Explain database indexing to:

1. A junior developer: Focus on *why* indexes matter and 
   basic CREATE INDEX syntax. Use analogies (book index).

2. A senior DBA: Discuss B-tree vs. hash indexes, partial 
   indexes, covering indexes, and query plan optimization.

3. A product manager: Explain how proper indexing affects 
   page load times and user experience metrics.
```

---

## Background Information

Provide context the model needs to give relevant responses:

### What to Include

```markdown
# Context
- Our company: B2B SaaS for healthcare
- Current situation: Migrating from monolith to microservices
- Tech stack: Python, FastAPI, PostgreSQL, Kubernetes
- Team size: 8 developers, 2 years experience average
- Constraints: HIPAA compliance required, limited DevOps capacity
```

### Context Categories

| Category | Example Content |
|----------|-----------------|
| **Company/Project** | Industry, size, product type |
| **Technical** | Stack, architecture, constraints |
| **Team** | Size, experience, capacity |
| **Situation** | Current problem, timeline, blockers |
| **History** | What's been tried, what failed |
| **Goals** | Short-term deliverable, long-term vision |

### Example: Full Context

```markdown
# Context
I'm building a customer support chatbot for an e-commerce company.

## Current State
- Using GPT-4o for responses
- RAG pipeline with product documentation
- 500 conversations/day, growing 20%/month
- Average response time: 3 seconds
- Customer satisfaction: 72% (target: 85%)

## Problem
Users complain responses are "too robotic" and "don't understand 
my specific situation."

## Constraints
- Budget: Can't switch to more expensive model
- Latency: Must stay under 5 seconds
- Integration: Must work with our existing Zendesk setup

# Task
Suggest 3 specific changes to our system prompt that would 
make responses feel more natural and contextual.
```

---

## Context Placement Strategies

Where you place context affects how the model uses it.

### System Message (Highest Priority)

Use for persistent behavior:

```javascript
const response = await openai.responses.create({
  model: "gpt-4o",
  instructions: `You are a customer service agent for TechCorp.
    
    Always be helpful, professional, and empathetic.
    Never reveal internal policies or make promises about refunds.
    Escalate to human agents for: refund requests over $100, 
    legal threats, or harassment.`,
  input: userMessage
});
```

### Developer Message

Use for task-specific instructions:

```javascript
const messages = [
  {
    role: "developer",
    content: `Today is ${new Date().toISOString()}.
      
      Current user: Premium subscriber since 2022.
      Recent orders: 3 in last 30 days.
      Open tickets: 1 (delivery delay, logged 2 days ago).
      
      Handle their inquiry with this context in mind.`
  },
  {
    role: "user", 
    content: "Where's my order?"
  }
];
```

### Context at End (Long Documents)

For long context, place instructions after:

```markdown
<document>
[10,000 words of documentation here]
</document>

Based on the documentation above, answer this question:
What are the rate limits for the /users endpoint?
```

> **🤖 AI Context:** Models process long contexts better when the question comes after the reference material. This is because attention mechanisms can "look back" at the document more effectively.

---

## Combining Role and Context

### Template

```markdown
# Role
You are a [role] with expertise in [specialization].

# Context
[Situation description]
[Relevant background]
[Constraints and requirements]

# Audience
You're speaking to [audience description].
They need [what they need from this interaction].

# Task
[Specific request]

# Output Format
[How to structure the response]
```

### Complete Example

```markdown
# Role
You are a senior security engineer specializing in web 
application security and OAuth 2.0 implementations.

# Context
We're building a SaaS application that will handle sensitive 
financial data. We need to implement user authentication 
using OAuth 2.0 with Google and Microsoft as identity providers.

Tech stack: Next.js, Node.js backend, PostgreSQL
Deployment: AWS (ECS, RDS)
Compliance: SOC 2 Type II certification in progress

# Audience
The development team (2 senior, 3 mid-level developers).
They understand OAuth basics but haven't implemented it 
in a production environment with compliance requirements.

# Task
Create a security checklist for our OAuth implementation 
covering the most critical items.

# Output Format
Numbered list, grouped by category (Configuration, Token 
Handling, Session Management, Logging). Include *why* each 
item matters for SOC 2 compliance.
```

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Overly generic role ("You are helpful") | Specific expertise and personality |
| No context about situation | Include relevant background |
| Assuming model knows your stack | State technologies explicitly |
| Forgetting audience | Specify who you're communicating with |
| Role contradicts task | Ensure role aligns with what you're asking |

---

## Hands-on Exercise

### Your Task

Write a role and context setup for an AI that will help users debug JavaScript code in a learning platform.

### Requirements

1. Define the role with specific expertise
2. Set the appropriate teaching style (encouraging, Socratic)
3. Specify the audience (learners, skill level)
4. Include context about the platform
5. Add constraints (what the AI should NOT do)

<details>
<summary>✅ Solution</summary>

```markdown
# Role
You are a friendly JavaScript tutor named "Debug Buddy" 
with expertise in:
- JavaScript ES6+ and modern best practices
- Common beginner mistakes and misconceptions
- Clear, encouraging explanations

# Teaching Style
- Use the Socratic method: Ask guiding questions before 
  giving answers
- Celebrate small wins and normalize making mistakes
- Explain *why* code doesn't work, not just how to fix it
- Connect concepts to real-world analogies when helpful

# Audience
Beginner to intermediate JavaScript learners (1-6 months 
of experience). They're learning on their own, often 
frustrated, and may lack CS fundamentals.

# Platform Context
This is an interactive coding platform where users:
- Write code in a browser-based editor
- See console output and error messages
- Can share their code with you for review

# Constraints
DO:
- Encourage experimentation
- Suggest console.log for debugging
- Point out what they did RIGHT before what's wrong

DO NOT:
- Give complete solutions immediately
- Use advanced concepts without explaining them
- Be condescending about "obvious" mistakes
- Write more than 20 lines of code at once

# Response Format
1. Acknowledge what they're trying to do
2. Identify the issue with a guiding question
3. Provide a hint if they're stuck
4. Celebrate when they figure it out
```

</details>

---

## Summary

✅ Roles activate specific expertise and communication styles

✅ Match expertise level to the complexity of your task

✅ Always specify your audience and their knowledge level

✅ Provide relevant background context for better responses

✅ Place context strategically based on prompt length

**Next:** [Instruction Ordering](./03-instruction-ordering.md)

---

## Further Reading

- [Anthropic: Give Claude a Role](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/system-prompts)
- [OpenAI: Message Roles](https://platform.openai.com/docs/guides/prompt-engineering)

---

<!-- 
Sources Consulted:
- OpenAI Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
- Anthropic System Prompts: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/system-prompts
- Google Gemini Prompting: https://ai.google.dev/gemini-api/docs/prompting-strategies
-->
