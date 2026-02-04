---
title: "Ambiguous Instructions"
---

# Ambiguous Instructions

## Introduction

Ambiguity is the silent killer of prompt engineering. When instructions can be interpreted multiple ways, models make their best guess—which may not match your intent. This lesson teaches you to identify ambiguity before it causes problems and apply clarification techniques that eliminate interpretation variance.

> **🔑 Key Insight:** If you can read your instruction and imagine two reasonable people interpreting it differently, the model will also vary its interpretation.

### What We'll Cover

- Recognizing ambiguous language patterns
- Systematic clarification techniques
- Edge case testing for hidden ambiguity
- Using examples to eliminate interpretation variance

### Prerequisites

- [Common Pitfalls Overview](./00-common-pitfalls-overview.md)
- [Fundamentals of Effective Prompts](../01-fundamentals-of-effective-prompts/00-fundamentals-overview.md)

---

## What Makes Instructions Ambiguous?

Ambiguity occurs when an instruction has multiple valid interpretations. Models don't ask for clarification—they pick one interpretation and proceed.

### The Three Types of Ambiguity

```
┌──────────────────────────────────────────────────────────────┐
│                    TYPES OF AMBIGUITY                         │
└──────────────────────────────────────────────────────────────┘

   LEXICAL                STRUCTURAL            REFERENTIAL
   ────────               ──────────            ───────────
   Same word,             Sentence can be       Unclear what
   multiple meanings      parsed multiple ways  pronoun refers to

   "Summarize this        "I saw the man with   "The system sends
   briefly"               the telescope"        it to the server"

   How brief?             Who has telescope?    What is "it"?
   100 words? 1 sentence? Man or speaker?       The data? Error?
```

| Ambiguity Type | Example | Problem |
|----------------|---------|---------|
| **Lexical** | "Make it better" | "Better" how? Faster? Cleaner? Shorter? |
| **Structural** | "Parse the JSON and XML files" | Parse JSON-and-XML, or JSON-files and XML-files? |
| **Referential** | "Update it after processing" | What is "it"? The data? The state? The UI? |

---

## Common Ambiguous Patterns

### Pattern 1: Relative Terms Without Baselines

❌ **Ambiguous:**
```
Write a short summary of this article.
```

✅ **Clear:**
```
Write a 2-3 sentence summary of this article.
```

**Common relative terms that need quantification:**

| Vague Term | Ask Yourself | Clarification Examples |
|------------|--------------|------------------------|
| "short" | How many words/sentences? | "under 100 words", "2-3 sentences" |
| "detailed" | How much detail? | "include all parameters", "cover edge cases" |
| "quickly" | What time constraint? | "respond in under 5 seconds", "prioritize speed over completeness" |
| "simple" | Simple for whom? | "understandable by non-programmers", "no jargon" |
| "good" | What criteria? | "follows PEP 8", "handles errors", "is testable" |
| "important" | Important to whom/why? | "critical for security", "required for compliance" |
| "appropriate" | By what standard? | "professional tone", "suitable for children" |

### Pattern 2: Unstated Scope

❌ **Ambiguous:**
```
Review this code for issues.
```

✅ **Clear:**
```
Review this code for:
1. Security vulnerabilities (SQL injection, XSS)
2. Performance issues (N+1 queries, unnecessary loops)
3. Code style violations (PEP 8 for Python)

Ignore: Minor style preferences, documentation completeness
```

### Pattern 3: Implicit Assumptions

❌ **Ambiguous:**
```
Format the output as JSON.
```

✅ **Clear:**
```
Format the output as JSON with:
- camelCase keys
- ISO 8601 dates (e.g., "2025-01-15T10:30:00Z")
- null for missing values (not empty strings)
- 2-space indentation for readability
```

### Pattern 4: Conditional Logic Without Coverage

❌ **Ambiguous:**
```
If the user is logged in, show their profile. Otherwise, show the login page.
```

What about users who are logged in but have no profile? What if login status is unknown?

✅ **Clear:**
```
Based on authentication state:
- Authenticated with profile → Display profile page
- Authenticated without profile → Display profile creation wizard
- Not authenticated → Display login page
- Unknown state (loading) → Display skeleton loader

Default to login page if state cannot be determined.
```

---

## Clarification Techniques

### Technique 1: Explicit Definitions

Define terms that could be interpreted differently:

```
# Task: Generate "professional" email responses

## Definition of "professional" for this task:
- Formal greeting (Dear/Hello [Name])
- No contractions (do not, will not, cannot)
- No emoji or casual language
- Sign-off with full name and title
- Maximum 3 paragraphs

## NOT considered professional in this context:
- Using "Hey" or first-name-only greetings
- Exclamation points (except one maximum)
- Casual phrases ("No worries", "Sounds good")
```

### Technique 2: Boundary Specification

Define what's in and out of scope:

```
## Scope

### Include:
- User-facing error messages
- Form validation feedback
- Success confirmations

### Exclude:
- Server logs
- Debug output
- Developer console messages

### Edge Cases:
- Timeout errors → Treat as user-facing, include
- Rate limiting messages → Include with friendly phrasing
- Maintenance notices → Include, mark as temporary
```

### Technique 3: Example-Based Clarification

Show don't tell—examples eliminate interpretation variance:

```
## Task: Rewrite sentences in active voice

### Examples:

Input: "The report was written by the team."
Output: "The team wrote the report."

Input: "Mistakes were made in the calculation."
Output: "Someone made mistakes in the calculation."

Input: "The ball was kicked by the player."
Output: "The player kicked the ball."

Note: When the actor is unknown (example 2), use "someone" 
rather than inventing a specific actor.
```

### Technique 4: Enumerated Options

When there are specific valid choices, list them:

```
## Response format

Use EXACTLY one of these classifications:

- APPROVE: Request meets all criteria
- REJECT: Request violates policy
- ESCALATE: Request requires human review
- INCOMPLETE: Request is missing required information

Do not use synonyms or variations. Output must match 
one of these four words exactly.
```

---

## Edge Case Testing

### Why Edge Cases Matter

Edge cases reveal hidden ambiguity. Test your prompts with unusual inputs before production.

```
┌─────────────────────────────────────────────────────────────┐
│              EDGE CASE TESTING FRAMEWORK                     │
└─────────────────────────────────────────────────────────────┘

Standard Test       Edge Case Test       Adversarial Test
────────────────    ────────────────     ─────────────────
Normal input        Unusual but valid    Deliberately tricky
Expected behavior   Boundary behavior    Breaking attempts

"Summarize this     "Summarize this      "Summarize this
 article about       article that is      article that 
 climate change"     just a headline"     contradicts itself"
```

### Edge Case Categories

| Category | What to Test | Example Input |
|----------|--------------|---------------|
| **Empty/null** | Missing input | Empty string, null, undefined |
| **Minimal** | Smallest valid input | Single word, single character |
| **Maximal** | Largest reasonable input | Very long text, many items |
| **Boundary** | At exact limits | Exactly 100 chars if limit is 100 |
| **Type edge** | Different valid types | Integer where float expected |
| **Format edge** | Unusual formatting | Unicode, mixed case, special chars |
| **Semantic edge** | Unusual meaning | Sarcasm, irony, contradictions |

### Testing Prompt for Ambiguity

```python
def test_prompt_for_ambiguity(prompt: str, test_inputs: list) -> dict:
    """
    Run the same prompt multiple times with same input
    to check for interpretation variance.
    """
    results = {}
    
    for test_input in test_inputs:
        # Run same prompt 5 times with temperature > 0
        outputs = [run_prompt(prompt, test_input) for _ in range(5)]
        
        # Check consistency
        unique_structures = count_unique_structures(outputs)
        
        results[test_input] = {
            "outputs": outputs,
            "unique_count": unique_structures,
            "is_ambiguous": unique_structures > 1
        }
    
    return results
```

### Edge Case Test Examples

**For a "summarization" prompt:**

| Edge Case | Input | What It Tests |
|-----------|-------|---------------|
| Empty | `""` | How to handle no content |
| Single sentence | `"Hello."` | Summarizing something already short |
| List format | `"1. First\n2. Second\n3. Third"` | Summarizing non-prose |
| Multiple topics | Mixed article | Maintaining focus or covering all |
| Contradictory | Article with conflicting claims | How to handle inconsistency |

**For a "classification" prompt:**

| Edge Case | Input | What It Tests |
|-----------|-------|---------------|
| Between categories | Ambiguous input | Tie-breaking behavior |
| No valid category | Unclassifiable input | Default/error handling |
| Multiple categories | Input fitting many | Single vs. multi-label |
| Sarcastic | Opposite meaning | Semantic understanding |

---

## Real-World Example: Fixing Ambiguous Prompts

### Case Study: Product Description Generator

**Original (Ambiguous):**
```
Write a product description that's engaging and informative.
Include key features and benefits.
```

**Problems identified:**

1. "Engaging" is subjective—what makes it engaging?
2. "Informative" is vague—how much information?
3. "Key features"—all features or selection?
4. Length unspecified
5. Tone unspecified
6. Format unspecified

**Improved (Clear):**
```
Write a product description with these specifications:

## Format
- Headline (5-10 words, benefit-focused)
- Opening hook (1 sentence, create urgency/desire)
- Feature-benefit bullets (3-5 items)
- Closing CTA (1 sentence)

## Tone
- Confident but not pushy
- Use "you/your" (customer-focused)
- Active voice
- No superlatives ("best", "greatest") unless provable

## Feature Selection
Include only features that:
- Differentiate from competitors, OR
- Solve a common customer pain point, OR
- Justify the price point

## Length
- Total: 100-150 words
- Each bullet: 10-20 words

## Example Output:
---
Headline: Never Miss a Moment with 72-Hour Battery Life

Tired of charging your smartwatch every night? The X200 
keeps pace with your life.

• All-day heart monitoring catches irregularities before 
  you notice symptoms
• Water-resistant to 50m—swim, shower, and sweat worry-free
• GPS tracking accurate to 3 meters for precise run mapping
• Sleep analysis with personalized improvement suggestions

Start your 30-day free trial today.
---
```

### Testing the Improvement

| Test | Ambiguous Version | Clear Version |
|------|-------------------|---------------|
| 5 runs same input | 5 different structures | Same structure |
| Bullet count | 2-8 bullets | 3-5 bullets |
| Length | 50-300 words | 100-150 words |
| Tone | Varied | Consistently customer-focused |

---

## Common Mistakes When Clarifying

### Mistake 1: Over-Clarifying Simple Tasks

❌ **Over-engineered:**
```
Add two numbers together. The numbers will be integers 
between -2147483648 and 2147483647. Return the result as 
an integer. Handle overflow by... [200 more words]
```

✅ **Appropriate:**
```
Add two numbers. Return the sum.
```

> **Tip:** Match clarification level to task complexity. Simple tasks need simple prompts.

### Mistake 2: Clarifying the Wrong Thing

❌ **Misplaced focus:**
```
Write Python code.
Use 4 spaces for indentation.
Use snake_case for variables.
Use type hints.
Use docstrings.

# What the code should do: not specified
```

✅ **Proper focus:**
```
Write a Python function that:
- Takes a list of integers
- Returns the two numbers that sum to a target
- Raises ValueError if no solution exists

Code style: Follow PEP 8 conventions.
```

### Mistake 3: Clarifying with More Ambiguity

❌ **Ambiguity spiral:**
```
Write a short summary.

Clarification: Make it brief but comprehensive.

Further clarification: Cover the main points without 
going into too much detail.
```

✅ **Concrete clarification:**
```
Write a summary:
- 2-3 sentences
- Cover: main argument, key evidence, conclusion
- Exclude: examples, quotes, tangential points
```

---

## Hands-on Exercise

### Your Task

Take this ambiguous prompt and rewrite it with full clarity:

**Original prompt:**
```
Review this code and provide feedback on how to improve it.
```

### Requirements

1. Define what "review" means (scope)
2. Specify what types of "improvements" to look for
3. Define the feedback format
4. Add examples of good feedback
5. Specify what to exclude from review
6. Test with an edge case (e.g., empty code, perfect code)

### Expected Result

A clear, unambiguous prompt that would produce consistent feedback across multiple runs.

<details>
<summary>💡 Hints (click to expand)</summary>

- Consider: What categories of issues? (bugs, style, performance, security)
- Consider: How should feedback be structured? (list, prioritized, with fixes?)
- Consider: What if the code is already good?
- Consider: What if there are many issues—prioritize how?

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```
## Code Review Task

Review the provided code and provide actionable feedback.

### Review Scope

Analyze for these issue types (in priority order):
1. **Security**: Vulnerabilities, unsafe practices
2. **Bugs**: Logic errors, edge cases, null handling
3. **Performance**: Inefficiencies, O(n²) when O(n) possible
4. **Maintainability**: Readability, naming, structure

### Explicitly Exclude
- Personal style preferences
- Minor formatting (assume auto-formatter handles this)
- Documentation completeness
- Test coverage

### Feedback Format

For each issue found:
```
[PRIORITY] Category: Brief description
Location: File/function/line
Problem: What's wrong and why it matters
Fix: Specific code or approach to resolve
```

### Examples

Good feedback:
```
[HIGH] Security: SQL injection vulnerability
Location: user_service.py, get_user(), line 23
Problem: User input directly concatenated into query string
Fix: Use parameterized query: cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```

Bad feedback (avoid this):
```
Code could be better organized.
```

### Edge Cases

- If code is high quality with no issues: State "No significant 
  issues found" and optionally suggest one minor enhancement
- If many issues exist: List top 5 by priority, note "Additional 
  issues exist; address these first"
- If code is incomplete/won't run: Note this first, then review 
  what's present
```

</details>

---

## Summary

✅ Ambiguity occurs when instructions can be interpreted multiple ways
✅ Three types: lexical (word meaning), structural (parsing), referential (pronouns)
✅ Clarify using: explicit definitions, boundary specs, examples, enumerated options
✅ Test edge cases to reveal hidden ambiguity
✅ Match clarification depth to task complexity—don't over-engineer simple prompts

**Next:** [Conflicting Requirements](./02-conflicting-requirements.md)

---

## Further Reading

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) - Strategies for clear instructions
- [Google Gemini Prompt Strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies) - Specificity techniques
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/) - Claude-specific guidance

---

<!-- 
Sources Consulted:
- OpenAI Prompt Engineering Guide: Message roles and clear instructions
- Anthropic Prompt Best Practices: Being explicit with instructions
- Google Gemini Strategies: Clear and specific instruction patterns
-->
