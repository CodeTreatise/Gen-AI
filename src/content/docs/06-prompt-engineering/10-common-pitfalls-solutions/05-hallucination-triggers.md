---
title: "Hallucination Triggers"
---

# Hallucination Triggers

## Introduction

Hallucinations—when models confidently generate false information—are the most dangerous prompt engineering failure. Unlike other pitfalls that produce wrong formats or inconsistent outputs, hallucinations appear authoritative while being factually wrong. This lesson teaches you to recognize prompts that trigger hallucinations and apply grounding techniques to minimize them.

> **🔑 Key Insight:** Models don't know what they don't know. They will generate plausible-sounding content whether or not it's accurate. It's your job to design prompts that minimize this risk.

### What We'll Cover

- Understanding why hallucinations occur
- Prompt patterns that trigger invention
- Grounding techniques to minimize fabrication
- Verification strategies for critical applications

### Prerequisites

- [Common Pitfalls Overview](./00-common-pitfalls-overview.md)
- [Token Waste](./04-token-waste.md)
- Basic understanding of how LLMs work

---

## Why Hallucinations Happen

### The Fundamental Problem

```
┌──────────────────────────────────────────────────────────────┐
│           WHY MODELS HALLUCINATE                              │
└──────────────────────────────────────────────────────────────┘

Models are trained to generate plausible continuations.
They optimize for: "What text would typically come next?"

NOT: "What is true?"
NOT: "What do I actually know?"
NOT: "Am I making this up?"

This creates a failure mode where the model generates
authoritative-sounding text that has no basis in fact.
```

### Hallucination Types

| Type | What It Is | Example |
|------|------------|---------|
| **Factual** | Wrong facts stated confidently | "The Eiffel Tower was built in 1920" |
| **Citation** | Made-up sources, papers, URLs | "According to Smith et al. (2023)..." that doesn't exist |
| **Capability** | Claims about what it can do | "I just checked your database and found..." |
| **Memory** | False claims about conversation | "As you mentioned earlier..." (you didn't) |
| **Temporal** | Wrong information about time | Current events from training data cutoff |
| **Technical** | Invalid code, APIs, configurations | Non-existent function calls |

---

## Prompt Patterns That Trigger Hallucination

### Pattern 1: Demanding Specifics Without Source

❌ **Triggers hallucination:**
```
What is the exact revenue of TechCorp for Q4 2024?
```

The model will often generate a plausible-sounding number rather than admitting it doesn't have this information.

✅ **Safer:**
```
If you have information about TechCorp's Q4 2024 revenue, provide it 
with the source. If you don't have this information, say so clearly.
```

### Pattern 2: Assuming the Model Has Access

❌ **Triggers hallucination:**
```
Look up the current price of Bitcoin.
```

The model may fabricate a current price rather than explain it can't access real-time data.

✅ **Safer:**
```
I need the current Bitcoin price. You cannot access real-time data.
Instead, explain how I can find this information.
```

### Pattern 3: Requesting Citations Without Sources

❌ **Triggers hallucination:**
```
Explain the health benefits of green tea with scientific citations.
```

Models will often generate plausible-sounding but non-existent paper citations.

✅ **Safer:**
```
Explain the health benefits of green tea based on your general knowledge.
Do NOT cite specific papers or studies—I will verify claims independently.
If you mention research, say "some studies suggest" rather than inventing citations.
```

### Pattern 4: Asking About Information Beyond Training

❌ **Triggers hallucination:**
```
What happened at the 2026 Olympics?
```

✅ **Safer:**
```
My knowledge cutoff is [date]. What happened at the 2026 Olympics?
If this is beyond your training data, say you don't have this information.
```

### Pattern 5: Implicit Requests for Fabrication

❌ **Triggers hallucination:**
```
Write a detailed biography of John Smith, a data scientist at Google.
```

Without verifying if this person exists, the model may create a fictional biography that sounds real.

✅ **Safer:**
```
Is John Smith, a data scientist at Google, someone you have information about?
If yes, provide verified details only.
If no, help me find how to verify this person's existence.
```

### Pattern 6: Technical Details Without Verification

❌ **Triggers hallucination:**
```
What are the API endpoints for the XYZ library?
```

Models frequently hallucinate function names, parameters, and API structures.

✅ **Safer:**
```
Describe the API structure of the XYZ library based on your training data.
Note: APIs change frequently. Mark any details you're uncertain about.
I will verify against current documentation before using.
```

---

## Grounding Techniques

### Technique 1: Provide Source Material

Give the model information to reference rather than invent:

```
Answer the following question using ONLY the provided context.
If the context doesn't contain the answer, say "Not found in provided context."

Context:
---
[paste relevant source material here]
---

Question: What was the company's revenue in Q4 2024?
```

### Technique 2: Explicit Uncertainty Instructions

Tell the model how to handle uncertainty:

```
For your response, categorize each claim:

VERIFIED: Information you are highly confident about
LIKELY: Information that is probably correct but may have nuances
UNCERTAIN: Information you are not sure about
UNKNOWN: Information you do not have

Example:
"The capital of France is Paris. [VERIFIED]
The population is approximately 2.1 million in the city proper. [LIKELY]
The current mayor's policies focus on... [UNCERTAIN - verify current data]"
```

### Technique 3: Explicit Knowledge Boundaries

Make the model's limitations part of the prompt:

```
## Your Knowledge Boundaries

You have information up to [training cutoff].
You CANNOT access:
- Real-time data
- The internet
- User's files or databases
- Current prices, scores, or live data

When asked about these, say:
"I cannot access [type of data]. Here's how you can find it: [alternative]"
```

### Technique 4: RAG-Based Grounding

For production systems, ground responses in retrieved documents:

```python
def grounded_response(query: str, documents: list[str]) -> str:
    prompt = f"""
    Answer the question using ONLY the information in the provided documents.
    
    Rules:
    - Quote relevant passages when possible
    - If information is not in documents, say "Not found in provided documents"
    - Do not add information beyond what's in the documents
    - Cite which document contains the information: [Doc 1], [Doc 2], etc.
    
    Documents:
    {format_documents(documents)}
    
    Question: {query}
    """
    return call_llm(prompt)
```

### Technique 5: The "Investigate First" Pattern

Tell the model to gather information before answering:

```
Before answering, follow this process:

1. INVESTIGATE: Read and understand all provided context
2. IDENTIFY: Note what information is available vs. missing
3. VERIFY: Check that your answer is supported by provided context
4. ANSWER: Respond using only verified information
5. ACKNOWLEDGE: Note any gaps or uncertainties

If you would need to guess or invent information, STOP and say what 
information is missing instead of guessing.
```

### Technique 6: Constrained Outputs for Critical Fields

For structured outputs where accuracy is critical:

```
Extract information from the document below.

For each field:
- FOUND: Extract exactly as written in document
- INFERRED: Derive from document (mark as inferred)
- MISSING: Field not present in document (use null)

CRITICAL: Never invent values. If a field is MISSING, return null.

Output format:
{
  "company_name": {"value": "...", "status": "FOUND|INFERRED|MISSING"},
  "revenue": {"value": ..., "status": "FOUND|INFERRED|MISSING"},
  ...
}
```

---

## Model-Specific Grounding Strategies

### OpenAI GPT-4/4o

```python
# Use developer message for grounding rules
messages = [
    {
        "role": "developer",
        "content": """
        You are grounded in provided context only.
        NEVER invent facts, citations, or statistics.
        When uncertain, express uncertainty rather than guessing.
        """
    },
    {
        "role": "user",
        "content": "[grounded context + question]"
    }
]
```

### Anthropic Claude

Claude's best practices emphasize explicit investigation:

```
## Investigation Protocol

ALWAYS investigate before answering:
- Read and understand relevant provided context
- Never speculate about information not provided
- If you need information you don't have, say so

NEVER:
- Invent citations or sources
- Claim to have checked external resources
- Generate specific numbers without source
```

### Google Gemini

Gemini documentation recommends explicit grounding instructions:

```
## Grounding Instructions

Treat the provided context as the ABSOLUTE LIMIT of truth.
Do not use information from your general training for factual claims.
All factual statements must be directly supported by the context.

If the context doesn't answer the question, respond:
"The provided context does not contain information about [topic]. 
Based on my general knowledge, [hedged statement if helpful], 
but please verify with authoritative sources."
```

---

## Verification Strategies

### Strategy 1: Self-Verification Prompting

Ask the model to check its own work:

```
After generating your response, verify each factual claim:

For each claim, ask yourself:
1. Do I actually know this, or am I pattern-matching?
2. Is this from the provided context or my general training?
3. Could this have changed since my training data?
4. Am I confident enough to include this without hedging?

Revise your response, adding hedging language or removing claims 
that fail verification.
```

### Strategy 2: Confidence Scoring

Request explicit confidence levels:

```
Provide your response with confidence scores:

Format each claim as:
[CLAIM] (Confidence: HIGH/MEDIUM/LOW)

HIGH: Directly stated in provided context or fundamental knowledge
MEDIUM: Inferred from context or common knowledge that may have nuances
LOW: Based on pattern matching, may not be accurate

For LOW confidence items, add: "Please verify: [what to check]"
```

### Strategy 3: External Verification Workflow

For production systems:

```python
def verified_response(query: str, context: str) -> dict:
    # Generate response with verification metadata
    response = llm.generate(
        prompt=f"""
        Answer this question using the context.
        Include verification metadata for each factual claim.
        
        Context: {context}
        Question: {query}
        """,
        output_format={
            "answer": "string",
            "claims": [{
                "statement": "string",
                "source": "context|training|inferred",
                "confidence": "high|medium|low",
                "verification_needed": "boolean"
            }]
        }
    )
    
    # Flag claims needing verification
    for claim in response["claims"]:
        if claim["verification_needed"]:
            flag_for_human_review(claim)
    
    return response
```

### Strategy 4: Adversarial Testing

Test your prompts with questions designed to trigger hallucination:

| Test Type | Example | What It Tests |
|-----------|---------|---------------|
| Nonexistent entity | "Tell me about XYZ Corp's CEO" (fake company) | Will it admit not knowing? |
| Future events | "What happened on [future date]?" | Will it say it doesn't know? |
| Fake citations | "Summarize the paper 'XYZ' by [made-up author]" | Will it fabricate content? |
| Current data | "What's the current stock price of Apple?" | Will it claim to access live data? |
| Specifics without source | "What's the exact population of [city]?" | Will it invent numbers? |

---

## Real-World Example: Research Assistant

### Before: Hallucination-Prone

```
You are a research assistant. Help users find information and 
cite relevant sources. Be thorough and academic in your responses.
```

**Problem:** Model will invent papers, citations, and statistics.

### After: Grounded Research Assistant

```
You are a research assistant. Your knowledge has limitations.

## What You Can Do
- Explain concepts from your training
- Describe general research findings (with hedging)
- Suggest search strategies and keywords
- Explain how to verify information

## What You Cannot Do
- Access current databases or papers
- Provide specific citations (you may misremember)
- Give current statistics or data
- Know about papers published after your training

## Response Format

For factual claims:
- "Research generally suggests..." (not "Studies show...")
- "As of my training data..." (acknowledge limits)
- "You may want to search for..." (suggest verification)

For citations:
- "You might find relevant work by searching [keywords]"
- "Check Google Scholar for recent work on [topic]"
- NEVER say "According to Smith (2023)..." unless you're certain

When uncertain:
- "I'm not certain about this—please verify"
- "This may have changed since my training"
- "I cannot confirm this is accurate"
```

---

## Hallucination Risk Assessment

Use this framework to assess hallucination risk in your prompts:

### High Risk (Add Strong Grounding)

- Requesting specific numbers, dates, statistics
- Asking for citations or sources
- Queries about current events or recent data
- Technical details (APIs, configurations, code)
- Information about specific individuals or companies
- Medical, legal, or financial advice

### Medium Risk (Add Some Grounding)

- General explanations of concepts
- Historical facts (widely known)
- Comparisons and analysis
- Code based on common patterns
- Well-established best practices

### Low Risk (Standard Prompting Okay)

- Creative writing (fiction is expected)
- Brainstorming and ideation
- Format conversions (structured to structured)
- Summarization of provided text
- Simple transformations

---

## Hands-on Exercise

### Your Task

The following prompt is designed to trigger hallucinations. Rewrite it to minimize hallucination risk.

**Original prompt:**
```
You are an expert financial analyst. Analyze the current market conditions
and provide specific investment recommendations. Include recent performance
data for recommended stocks, current P/E ratios, and cite relevant 
financial reports from major institutions.
```

### Requirements

1. Identify all hallucination triggers in the original
2. Rewrite with appropriate grounding
3. Add uncertainty handling
4. Specify what the model CAN and CANNOT do
5. Include a verification workflow for the user

<details>
<summary>💡 Hints (click to expand)</summary>

**Hallucination triggers to find:**
- "Current market conditions" - real-time data
- "Specific investment recommendations" - invites invention
- "Recent performance data" - needs live data
- "Current P/E ratios" - specific numbers change constantly
- "Cite relevant financial reports" - citation hallucination

**Grounding elements to add:**
- Knowledge boundaries
- What user should provide
- Hedging requirements
- Verification instructions

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**Hallucination triggers identified:**

| Trigger | Problem |
|---------|---------|
| "Current market conditions" | Cannot access real-time data |
| "Specific investment recommendations" | May invent companies/tickers |
| "Recent performance data" | Data changes constantly |
| "Current P/E ratios" | Specific numbers likely wrong |
| "Cite relevant financial reports" | Will fabricate citations |

**Rewritten prompt:**

```
## Financial Analysis Assistant

### Knowledge Boundaries
You have financial knowledge from training data up to [cutoff date].
You CANNOT access:
- Current stock prices or market data
- Real-time P/E ratios or financial metrics
- Recent financial reports or earnings calls
- Live news or market events

### What You CAN Help With
- Explain financial concepts and analysis frameworks
- Discuss general investment principles
- Analyze specific data the user provides
- Suggest what metrics to consider
- Help interpret financial statements (if provided)

### User Must Provide
For specific analysis, user should provide:
- Current financial data from authorized sources
- Recent performance metrics from brokerage/financial sites
- Specific reports they want interpreted

### Response Requirements

For all factual claims:
- State "Based on general financial principles..." not "Currently..."
- Never invent specific numbers (prices, ratios, percentages)
- If discussing specific companies, say "As of my training data..."

For recommendations:
- Frame as "Factors to consider..." not "You should buy..."
- Always include: "Verify current data before any decision"
- Note: "This is not financial advice—consult a licensed advisor"

### Verification Workflow
After your response, include:

---
VERIFY BEFORE ACTING:
- [ ] Check current data on [suggested sources: Bloomberg, Yahoo Finance]
- [ ] Confirm specific metrics are current
- [ ] Consult licensed financial advisor for personalized advice
---
```

**Why this works:**
- Clear boundaries prevent capability hallucination
- User provides current data—model doesn't invent it
- Hedging language required throughout
- Explicit verification workflow
- Appropriate disclaimers

</details>

---

## Summary

✅ Hallucinations occur because models optimize for plausibility, not truth
✅ Six trigger patterns: demanding specifics, assuming access, requesting citations, asking beyond training, implicit fabrication requests, unverified technical details
✅ Grounding techniques: provide sources, explicit uncertainty, knowledge boundaries, RAG, investigate-first pattern
✅ Different models benefit from model-specific grounding instructions
✅ Use verification workflows for high-stakes applications
✅ Assess hallucination risk and apply grounding proportionally

**Previous:** [Token Waste](./04-token-waste.md)

**Back to:** [Common Pitfalls Overview](./00-common-pitfalls-overview.md)

---

## Further Reading

- [Anthropic Claude Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/) - Hallucination minimization
- [Google Gemini Grounding](https://ai.google.dev/gemini-api/docs/prompting-strategies) - Context grounding techniques
- [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering) - Accuracy strategies

---

<!-- 
Sources Consulted:
- Anthropic: "Never speculate about code you have not opened", investigation protocol
- Google Gemini: Grounding prompt template, treating context as absolute truth
- OpenAI: Developer message priority for grounding rules
- Research on LLM hallucination patterns and mitigation
-->
