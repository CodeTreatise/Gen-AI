---
title: "Clarity and Specificity"
---

# Clarity and Specificity

## Introduction

Vague prompts produce vague outputs. The more precisely you describe what you want, the more likely you are to get exactly that. This lesson covers techniques for eliminating ambiguity and writing prompts that leave no room for misinterpretation.

### What We'll Cover

- Avoiding ambiguous language
- Using precise terminology
- Including concrete examples
- Defining scope and constraints
- Setting expected output length

### Prerequisites

- [Fundamentals Overview](./00-fundamentals-overview.md)

---

## Why Clarity Matters

Consider these two prompts:

```
❌ Vague: "Summarize this article"

✅ Clear: "Summarize this article in 3 bullet points, focusing on the 
main argument, key evidence, and conclusion. Each bullet should be 
one sentence."
```

The vague prompt could produce a 500-word summary or a 10-word one. The clear prompt specifies format, length, and focus.

---

## Avoiding Ambiguous Language

### Problem Words

Certain words are inherently ambiguous:

| Ambiguous | Better Alternative |
|-----------|-------------------|
| "good" | "follows PEP 8 style guidelines" |
| "short" | "under 100 words" |
| "professional" | "formal tone, no contractions, third person" |
| "simple" | "uses only basic JavaScript (no frameworks)" |
| "better" | "more readable, with descriptive variable names" |
| "fast" | "completes in under 100ms" |

### Examples

```
❌ "Write a good function"
✅ "Write a Python function that is type-annotated, has a docstring, 
   and follows PEP 8 naming conventions"

❌ "Make it shorter"
✅ "Reduce to 50 words while keeping the main points"

❌ "Be more professional"
✅ "Use formal business language, third person, no slang or contractions"
```

---

## Precise Terminology

Use domain-specific terms when appropriate:

```python
# ❌ Vague
"Fix the code so it handles errors better"

# ✅ Precise
"Add try-except blocks to handle FileNotFoundError and 
PermissionError. Log exceptions using the logging module 
at ERROR level. Re-raise unexpected exceptions."
```

For technical tasks, precision prevents misunderstanding:

| Vague | Precise |
|-------|---------|
| "database" | "PostgreSQL 15" |
| "modern JavaScript" | "ES2022+ syntax" |
| "RESTful" | "REST API with JSON responses, HTTP status codes" |
| "secure" | "input validation, parameterized queries, HTTPS" |
| "tested" | "unit tests with pytest, minimum 80% coverage" |

---

## Concrete Examples in Prompts

Examples communicate more than instructions. Show the model exactly what you want:

### Without Example

```
Convert the text to title case.
```

**Problem:** What about small words? Articles? Prepositions?

### With Example

```
Convert the text to title case.

Example:
Input: "the quick brown fox jumps over the lazy dog"
Output: "The Quick Brown Fox Jumps over the Lazy Dog"

Note: Lowercase articles (a, an, the) and prepositions (over, in, on) 
unless they start the title.
```

### Multiple Examples for Patterns

```
Classify the sentiment as Positive, Negative, or Neutral.

Examples:
- "I love this product!" → Positive
- "Terrible experience, never again." → Negative  
- "It arrived on Tuesday." → Neutral
- "Not bad, but not great either." → Neutral

Now classify:
- "This exceeded my expectations!"
```

---

## Defining Scope

Clearly state what's in scope and out of scope:

### Scope Definition

```markdown
# Task
Write a product description for the new wireless headphones.

# In Scope
- Key features (battery life, sound quality, comfort)
- Target audience (commuters, remote workers)
- Call to action

# Out of Scope
- Technical specifications
- Comparison to competitors
- Pricing information
```

### Boundary Setting

```
Analyze the code for security vulnerabilities.

Focus ONLY on:
- SQL injection
- XSS vulnerabilities
- Authentication issues

Do NOT analyze:
- Performance
- Code style
- General best practices
```

---

## Expected Output Length

Specify length explicitly:

| Approach | Example |
|----------|---------|
| **Word count** | "Respond in 50-75 words" |
| **Sentences** | "Answer in exactly 2 sentences" |
| **Paragraphs** | "Write 3 paragraphs of 100 words each" |
| **List items** | "Provide exactly 5 bullet points" |
| **Relative** | "Maximum 1/4 the length of the input" |

### Examples

```
# Too open-ended
"Explain quantum computing"

# Length-controlled
"Explain quantum computing in 3 sentences suitable for a 
high school student"

# Structured length
"Explain quantum computing:
- Definition: 1 sentence
- How it differs from classical: 2 sentences  
- Real-world application: 1 sentence"
```

---

## Combining Techniques

Here's a fully specified prompt using all techniques:

```markdown
# Task
Write a product announcement email for our new AI code assistant.

# Audience
Software developers who have signed up for our waitlist.

# Tone
Enthusiastic but professional. Technical but accessible.

# Structure
1. Subject line (under 50 characters, creates urgency)
2. Opening hook (1 sentence, reference their waitlist signup)
3. Key features (3 bullet points, each 15-20 words)
4. Call to action (1 sentence, link to get started)
5. Sign-off (casual, first name only)

# Constraints
- Total length: 150-200 words (excluding subject line)
- No jargon without explanation
- Include one specific metric (e.g., "saves 2 hours per day")

# Example bullet point format
"✨ Smart autocomplete — Suggests entire functions based on your 
comments, reducing boilerplate by 60%"
```

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using "etc." or "and so on" | List all items explicitly |
| Saying "be creative" | Specify the type of creativity wanted |
| "Make it better" | Define what "better" means |
| Assuming shared context | State all relevant background |
| Open-ended length | Specify word/sentence/item counts |

---

## Hands-on Exercise

### Your Task

Take this vague prompt and rewrite it with full clarity:

**Original:**
```
"Write something about our new app for social media"
```

### Requirements

1. Define the specific output (post, ad, thread?)
2. Specify the platform
3. Set tone and audience
4. Include length constraints
5. Add at least one example or format specification

<details>
<summary>💡 Hints</summary>

- What platform? (Twitter, LinkedIn, Instagram?)
- What's the app about?
- Who's the target audience?
- What's the goal? (awareness, signups, engagement?)
- How long should it be?

</details>

<details>
<summary>✅ Solution</summary>

```markdown
# Task
Write a LinkedIn post announcing our new AI meeting assistant app.

# Audience
Busy professionals and remote workers who attend 5+ meetings per week.

# Goal
Drive app store downloads with a clear value proposition.

# Tone
Professional but conversational. Use "you" language. Confident, not salesy.

# Structure
1. Hook question (addresses pain point)
2. Problem statement (2 sentences)
3. Solution introduction (1 sentence)
4. Key benefits (3 bullet points, emoji prefix)
5. Call to action with urgency
6. Relevant hashtags (max 3)

# Constraints
- Length: 150-200 words
- Include one specific stat (e.g., "saves 3 hours/week")
- No buzzwords like "game-changer" or "revolutionary"

# Example hook
"Spent another hour in a meeting that could've been an email?"
```

</details>

---

## Summary

✅ Replace vague words with measurable, specific terms

✅ Use domain-appropriate technical terminology

✅ Include concrete examples to demonstrate expected output

✅ Define scope explicitly (what's in, what's out)

✅ Always specify expected length and format

**Next:** [Role and Context Setting](./02-role-and-context-setting.md)

---

## Further Reading

- [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic: Be Clear and Direct](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/be-clear-and-direct)

---

<!-- 
Sources Consulted:
- OpenAI Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
- Anthropic Be Clear and Direct: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/be-clear-and-direct
- Google Gemini Prompting Strategies: https://ai.google.dev/gemini-api/docs/prompting-strategies
-->
