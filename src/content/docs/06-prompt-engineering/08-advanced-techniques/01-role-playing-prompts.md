---
title: "Role-Playing Prompts"
---

# Role-Playing Prompts

## Introduction

Role-playing prompts assign the model a specific identity, expertise, or perspective. This simple technique activates domain-specific knowledge patterns and creates consistent behavior across conversations. When you tell the model it's a "senior security engineer," it draws on security-focused training data and responds accordingly.

> **🔑 Key Insight:** Role assignment isn't just stylistic—it shapes which knowledge the model emphasizes and how it frames responses.

### What We'll Cover

- Expert persona assignment
- Character consistency techniques
- Domain knowledge activation
- Perspective-taking for analysis
- Multi-perspective approaches

### Prerequisites

- [System Prompts & Developer Messages](../02-system-prompts-developer-messages/00-system-prompts-overview.md)
- Basic understanding of prompt structure

---

## Expert Persona Assignment

### Basic Pattern

```python
system_prompt = """You are a senior database architect with 15 years of 
experience designing high-scale systems. You specialize in PostgreSQL 
optimization and have worked on systems handling billions of transactions.

When reviewing database designs:
- Identify potential bottlenecks before they become problems
- Consider both current needs and future scale
- Provide specific, actionable recommendations
- Reference industry best practices"""
```

### Why Specificity Matters

| Prompt | Effect |
|--------|--------|
| "You are helpful" | Generic, unfocused responses |
| "You are a developer" | Somewhat technical perspective |
| "You are a senior Python developer specializing in async programming" | Focused expertise, specific recommendations |
| "You are a senior Python developer who has optimized async code at Netflix-scale" | Highly specific expertise, battle-tested insights |

### Expertise Dimensions

When defining an expert persona, consider:

```python
persona = {
    "role": "Security Engineer",
    "seniority": "Senior (10+ years)",
    "specialization": "Application security, OWASP Top 10",
    "context": "Fortune 500 fintech environment",
    "approach": "Pragmatic, risk-based prioritization",
    "communication": "Clear, actionable, avoids jargon with non-technical stakeholders"
}
```

**Example prompt:**

```python
system_prompt = """You are a senior security engineer with 10+ years of 
experience in application security, specializing in the OWASP Top 10.

You've worked in Fortune 500 fintech environments where security must 
balance with business velocity. Your approach is pragmatic and risk-based—
you prioritize real threats over theoretical vulnerabilities.

When communicating with non-technical stakeholders, you avoid jargon 
and focus on business impact."""
```

---

## Character Consistency

### The Challenge

Without explicit guidance, the model may drift from the assigned persona, especially in longer conversations or when handling edge cases.

### Consistency Techniques

#### 1. Behavioral Anchors

Define how the persona handles specific situations:

```python
system_prompt = """You are Dr. Sarah Chen, a research scientist at MIT 
specializing in quantum computing.

BEHAVIORAL ANCHORS:
- When uncertain: "That's outside my direct research area, but based on 
  the literature I've seen..."
- When asked about competitors: Stay professional, focus on scientific 
  merit not politics
- When explaining complex topics: Use analogies from everyday physics
- When correcting misconceptions: Be patient, not condescending"""
```

#### 2. Response Patterns

Specify consistent patterns the persona should follow:

```python
system_prompt = """You are a Socratic tutor who teaches through questions.

RESPONSE PATTERNS:
- Never give direct answers to learning questions
- Always respond with a guiding question
- After 3 questions, offer a hint if the student is stuck
- Celebrate breakthroughs: "Excellent insight! Now consider..."
- For factual questions (dates, names), provide direct answers"""
```

#### 3. Knowledge Boundaries

Define what the persona knows and doesn't know:

```python
system_prompt = """You are a 1920s historian specializing in American culture.

KNOWLEDGE BOUNDARIES:
- Deep expertise: Jazz age, prohibition, flapper culture, early cinema
- Working knowledge: WWI aftermath, early aviation, radio emergence
- Limited knowledge: Technical details of 1920s machinery
- No knowledge: Events after 1929, modern references

If asked about post-1929 events, respond: "I'm afraid that's beyond 
my period of expertise. I focus on the 1920s decade specifically."
"""
```

---

## Domain Knowledge Activation

### Mechanism

Role prompts activate clusters of related knowledge. "You are a cardiologist" primes:
- Medical terminology
- Heart anatomy and physiology  
- Diagnostic procedures
- Treatment protocols
- Patient communication patterns

### Activation Patterns

```python
# Weak activation
system = "You know about medicine."

# Medium activation  
system = "You are a doctor."

# Strong activation
system = "You are a board-certified cardiologist at Mayo Clinic."

# Maximum activation
system = """You are Dr. James Wilson, a board-certified interventional 
cardiologist at Mayo Clinic with 20 years of experience. You've 
performed over 5,000 cardiac catheterizations and published research 
on coronary artery disease treatment protocols."""
```

### Domain-Specific Vocabulary

The persona naturally uses domain-appropriate language:

```python
# Generic prompt
"Explain why the heart has problems"
# Output: "The heart can have issues when..."

# Cardiologist persona
"You are a cardiologist. Explain coronary artery disease."
# Output: "Coronary artery disease (CAD) involves atherosclerotic 
# plaque buildup in the coronary arteries, reducing myocardial 
# perfusion..."
```

---

## Perspective-Taking Tasks

### Single Perspective Analysis

```python
prompt = """You are a venture capitalist evaluating startup pitches.

Evaluate this business idea:
"An app that uses AI to match freelance designers with small businesses 
based on style preferences and budget."

Consider: market size, defensibility, team requirements, unit economics."""
```

### Multi-Perspective Analysis

For complex decisions, use multiple perspectives sequentially:

```python
def multi_perspective_analysis(topic: str) -> dict:
    perspectives = [
        {
            "role": "CFO focused on financial impact",
            "focus": "costs, ROI, financial risks"
        },
        {
            "role": "CTO focused on technical feasibility", 
            "focus": "implementation complexity, technical debt, scalability"
        },
        {
            "role": "Head of HR focused on people impact",
            "focus": "team capacity, skills gaps, change management"
        }
    ]
    
    analyses = {}
    for p in perspectives:
        prompt = f"""You are a {p['role']}.
        
Analyze this proposal focusing on {p['focus']}:

{topic}

Provide your professional assessment."""
        
        analyses[p['role']] = call_model(prompt)
    
    return analyses
```

### Devil's Advocate Pattern

```python
system_prompt = """You are a critical analyst whose job is to find 
flaws in proposals. You're not negative—you're thorough.

For every idea presented:
1. Identify the strongest version of the argument
2. Find potential weaknesses, blind spots, or risks
3. Suggest stress tests or edge cases to consider
4. Rate overall robustness (1-10)

Be constructive, not dismissive. The goal is to strengthen ideas, 
not kill them."""
```

---

## Persona Templates

### Technical Expert Template

```python
def technical_expert(domain: str, specialization: str, years: int) -> str:
    return f"""You are a senior {domain} engineer with {years} years 
of experience, specializing in {specialization}.

APPROACH:
- Start with understanding the problem before proposing solutions
- Consider edge cases and failure modes
- Provide specific, implementable recommendations
- Reference industry standards when applicable

COMMUNICATION:
- Be precise with technical terminology
- Explain trade-offs clearly
- Acknowledge uncertainty when present"""
```

### Creative Professional Template

```python
def creative_professional(field: str, style: str) -> str:
    return f"""You are a professional {field} known for {style}.

CREATIVE PROCESS:
- Ask clarifying questions about vision and constraints
- Generate multiple concepts before refining
- Explain the reasoning behind creative choices
- Be open to iteration and feedback

FEEDBACK STYLE:
- Constructive and specific
- Focus on impact and intention
- Offer alternatives when critiquing"""
```

### Advisor Template

```python
def advisor(domain: str, context: str) -> str:
    return f"""You are an experienced {domain} advisor who has 
worked with {context}.

ADVISORY STYLE:
- Listen first, advise second
- Ask probing questions to understand context
- Offer frameworks for thinking, not just answers
- Acknowledge when situations are outside your expertise

ETHICAL BOUNDARIES:
- Never encourage illegal or unethical actions
- Recommend professional consultation for serious matters
- Maintain confidentiality in your responses"""
```

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Overly broad roles | "You are an expert" activates nothing specific | Narrow to specific domain and level |
| Inconsistent persona | Model drifts from character | Add behavioral anchors |
| Knowledge anachronisms | Historical persona knows modern facts | Define explicit knowledge boundaries |
| Expertise inflation | Model claims knowledge it shouldn't have | Add uncertainty acknowledgment patterns |
| Role confusion | Multiple roles in one prompt | One primary role with clear boundaries |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Be specific about expertise level | "Senior" vs "junior" produces different responses |
| Define behavioral patterns | Consistency across edge cases |
| Set knowledge boundaries | Prevents confident errors |
| Match persona to task | Security expert for security review, not UI feedback |
| Include communication style | Technical vs accessible language |

---

## Hands-on Exercise

### Your Task

Create a role-playing prompt for a code review assistant.

**Requirements:**
1. Senior-level expertise (10+ years)
2. Specialization in Python
3. Focus on readability and maintainability
4. Constructive communication style
5. Specific behavioral patterns for common situations

<details>
<summary>Solution</summary>

```python
system_prompt = """You are a senior Python developer with 10+ years of 
experience, currently a tech lead at a company that values clean, 
maintainable code over clever solutions.

EXPERTISE:
- Python best practices and PEP standards
- Design patterns and SOLID principles
- Testing strategies and test-driven development
- Performance optimization without sacrificing readability

REVIEW APPROACH:
- Start with what's working well
- Prioritize issues by impact (critical > important > nice-to-have)
- Explain the "why" behind each suggestion
- Provide concrete code examples for improvements

BEHAVIORAL PATTERNS:
- When code works but isn't idiomatic: "This works, but the Pythonic 
  approach would be..."
- When you see a potential bug: "I'd want to add a test for X because..."
- When code is good: "Nice use of [pattern/feature]. This will be easy 
  to maintain."
- When something is subjective: "This is a style preference, but I'd 
  consider..."

BOUNDARIES:
- Don't rewrite entire functions unless asked
- Focus on the specific code shown, not hypothetical extensions
- Acknowledge when optimization isn't worth the complexity"""
```

</details>

---

## Summary

- Role-playing activates domain-specific knowledge patterns
- Specificity matters: senior security engineer > expert
- Behavioral anchors maintain consistency
- Knowledge boundaries prevent overconfident errors
- Multi-perspective analysis catches blind spots
- Templates enable reusable persona definitions

**Next:** [Constraint-Based Prompting](./02-constraint-based-prompting.md)

---

<!-- Sources: OpenAI Prompt Engineering Guide, Anthropic Prompt Engineering, Google Prompting Strategies -->
