---
title: "Over-Constraining and Under-Specifying"
---

# Over-Constraining and Under-Specifying

## Introduction

Finding the right level of specification is one of prompt engineering's core challenges. Too many constraints confuse models and limit their capabilities. Too few constraints leave models guessing at your intent. This lesson teaches you to recognize both extremes and find the productive middle ground.

> **🔑 Key Insight:** The goal isn't maximum or minimum constraints—it's the *right* constraints. Essential boundaries without unnecessary restrictions.

### What We'll Cover

- Signs your prompt is over-constrained
- Signs your prompt is under-specified
- Techniques for finding the balance
- The "essential constraints only" principle

### Prerequisites

- [Common Pitfalls Overview](./00-common-pitfalls-overview.md)
- [Conflicting Requirements](./02-conflicting-requirements.md)

---

## Over-Constraining: Too Many Rules

### What Over-Constraining Looks Like

```
┌──────────────────────────────────────────────────────────────┐
│              OVER-CONSTRAINING SPECTRUM                       │
└──────────────────────────────────────────────────────────────┘

UNDER-CONSTRAINED    BALANCED        OVER-CONSTRAINED
       ◄────────────────────────────────────────────►
       
"Write something"    "Write a 100-   "Write exactly 147 words.
                      word summary     Start with 'The'.
                      covering the     Use no contractions.
                      main points"     Include 3 statistics.
                                       No passive voice.
                                       Exactly 4 paragraphs.
                                       Each paragraph 2-4 sentences.
                                       First word of each paragraph
                                       must be different.
                                       End with a question.
                                       No words over 3 syllables.
                                       Include the word 'therefore'
                                       exactly twice..."
```

### Signs Your Prompt Is Over-Constrained

| Symptom | What It Looks Like | Cause |
|---------|---------------------|-------|
| **Refusals** | "I cannot fulfill this request" | Impossible constraint combination |
| **Confused outputs** | Incoherent or contradictory responses | Too many rules to track |
| **Literal interpretation** | Follows letter, not spirit of request | Over-reliance on explicit rules |
| **Degraded quality** | Stilted, unnatural writing | Creativity suppressed by rules |
| **Partial compliance** | Follows some rules, ignores others | Constraint overload |

### Over-Constraining Patterns

#### Pattern 1: Micro-Managing Format

❌ **Over-constrained:**
```
Format your response as follows:
- First line: Title in exactly 5 words
- Second line: Blank
- Lines 3-5: Introduction (exactly 3 lines)
- Line 6: Blank
- Lines 7-12: Main content (exactly 6 lines)
- Line 13: Blank  
- Lines 14-15: Conclusion (exactly 2 lines)
- Each line must be 50-60 characters
- No line may start with "The" or "A"
- Every third line must contain a number
```

✅ **Appropriately constrained:**
```
Format your response with:
- Title
- Brief introduction (1-2 sentences)
- Main content (2-3 paragraphs)
- Conclusion (1-2 sentences)

Use clear, scannable structure with logical flow.
```

#### Pattern 2: Exhaustive Negative Rules

❌ **Over-constrained:**
```
Do NOT use:
- Passive voice
- Contractions
- Exclamation marks
- Questions
- The words "very", "really", "just", "actually"
- Sentences over 20 words
- More than 2 commas per sentence
- Semicolons
- Em dashes
- Parenthetical asides
- Adverbs ending in "-ly"
- Starting sentences with "It" or "There"
```

✅ **Appropriately constrained:**
```
Write in clear, direct prose:
- Active voice preferred
- Concise sentences (aim for 15-20 words average)
- Formal but accessible tone
```

> **Warning:** Lists of "don'ts" are harder to follow than positive instructions. Models pay more attention to what you tell them TO do.

#### Pattern 3: Redundant Constraints

❌ **Redundant:**
```
Respond in JSON format.
Your response must be valid JSON.
Make sure the output is parseable JSON.
Do not include any text outside the JSON.
The JSON must be properly formatted.
Ensure all keys are quoted strings.
Arrays must use square brackets.
```

✅ **Sufficient:**
```
Respond with valid JSON only. No other text.
```

---

## Under-Specifying: Not Enough Guidance

### What Under-Specifying Looks Like

```
┌──────────────────────────────────────────────────────────────┐
│          UNDER-SPECIFICATION EXAMPLES                         │
└──────────────────────────────────────────────────────────────┘

PROMPT                          MODEL'S DILEMMA
──────                          ────────────────
"Improve this code"             Improve HOW? Performance? 
                                Readability? Security?

"Write about dogs"              What aspect? How long? 
                                What audience? What tone?

"Fix the bug"                   What's the bug? What's
                                correct behavior?

"Make it better"                Better by what measure?
```

### Signs Your Prompt Is Under-Specified

| Symptom | What It Looks Like | Cause |
|---------|---------------------|-------|
| **Wrong assumptions** | Model guesses wrong context | Missing background |
| **Inconsistent outputs** | Different structure each time | No format specified |
| **Unexpected scope** | Response too long/short | No length guidance |
| **Wrong audience** | Too technical or too simple | No audience specified |
| **Missing elements** | Key information not included | Requirements unclear |

### Under-Specification Patterns

#### Pattern 1: Missing Context

❌ **Under-specified:**
```
Analyze this sales data.
```

✅ **Properly specified:**
```
Analyze this sales data for a quarterly business review.

Context:
- Audience: Senior leadership (non-technical)
- Focus: Revenue trends and regional performance  
- Time period: Q4 2024 vs Q4 2023
- Our targets: 15% YoY growth, $2M monthly revenue

Deliverable: 3-5 key insights with implications
```

#### Pattern 2: Assumed Knowledge

❌ **Under-specified:**
```
Refactor this to use the new pattern.
```

✅ **Properly specified:**
```
Refactor this code from class-based components to React hooks.

Current: Class component with lifecycle methods
Target: Functional component with useState, useEffect

Preserve:
- All existing functionality
- Component prop interface
- Error handling behavior
```

#### Pattern 3: Vague Success Criteria

❌ **Under-specified:**
```
Write a good product description.
```

✅ **Properly specified:**
```
Write a product description that:
- Converts browsers to buyers (CTA-focused)
- Highlights 3 key differentiators from competitors
- Addresses the main objection: "Is it worth the price?"
- Targets: Busy professionals, 35-50, value quality over cost
- Length: 100-150 words
- Tone: Confident, not salesy
```

---

## Finding the Balance

### The "Essential Constraints Only" Principle

```
┌──────────────────────────────────────────────────────────────┐
│            ESSENTIAL vs NON-ESSENTIAL CONSTRAINTS             │
└──────────────────────────────────────────────────────────────┘

ASK: "If I removed this constraint, would the output be wrong?"

YES → Essential. Keep it.
NO  → Non-essential. Consider removing.

EXAMPLE:

Constraint                            Essential?
─────────────────────────────────────────────────
"Return valid JSON"                   YES - parsing would fail
"Use camelCase keys"                  MAYBE - depends on system
"2-space indentation"                 NO - cosmetic preference
"No trailing commas"                  YES - some parsers fail
"Include 'timestamp' field"           YES - required by schema
"Fields in alphabetical order"        NO - typically irrelevant
```

### The Constraint Evaluation Framework

For each constraint in your prompt, evaluate:

| Question | If YES | If NO |
|----------|--------|-------|
| Does this prevent incorrect output? | Keep it | Consider removing |
| Does this reduce ambiguity? | Keep it | Consider removing |
| Does this conflict with other constraints? | Resolve or remove | Keep evaluating |
| Is this a personal preference? | Remove it | Keep evaluating |
| Could the model reasonably infer this? | Maybe remove | Probably keep |

### Progressive Constraint Building

Start minimal, add constraints only when needed:

**Step 1: Minimal viable prompt**
```
Summarize this article.
```

**Step 2: Test output, identify issues**
- Output was 500 words (too long)
- Output missed the main conclusion
- Output was too technical for audience

**Step 3: Add only necessary constraints**
```
Summarize this article:
- Length: 50-75 words
- Must include: main thesis and conclusion
- Audience: General reader (explain jargon)
```

**Step 4: Test again**
- Now appropriate length ✓
- Includes key points ✓
- Accessible language ✓

**Stop here.** Don't add more constraints just because you could.

---

## Constraint Audit Checklist

Use this checklist to review existing prompts:

### Remove If Present:

| Pattern | Example | Why Remove |
|---------|---------|------------|
| Redundant rules | "Be clear. Use clear language. Clarity is important." | One statement sufficient |
| Cosmetic preferences | "Use 4 spaces, not tabs" | Unless technically required |
| Unenforceable rules | "Be creative but not too creative" | Vague, can't verify |
| Conflicting with model nature | "Never use patterns from training" | Impossible |
| Obvious instructions | "Use correct grammar" | Model defaults to this |

### Add If Missing:

| What's Needed | Example | Why Add |
|---------------|---------|---------|
| Output format | "Return as JSON" / "Use bullet points" | Ensures parseability/structure |
| Length bounds | "100-200 words" | Prevents too long/short |
| Critical inclusions | "Must mention: X, Y, Z" | Ensures required elements |
| Audience context | "For senior engineers" | Calibrates complexity |
| Edge case handling | "If X is missing, return null" | Prevents guessing |

---

## Practical Examples

### Example 1: Code Generation

**Over-constrained:**
```
Write a Python function.
- Use exactly 4 spaces for indentation
- Maximum line length 79 characters
- Use snake_case for all names
- Include type hints for all parameters
- Include type hint for return value  
- Add a docstring with Google style format
- Docstring must include Args, Returns, Raises sections
- Use f-strings not .format() or %
- No global variables
- No mutable default arguments
- Use walrus operator where applicable
- Prefer list comprehensions over map/filter
- Add inline comments for complex logic
- Function must be pure (no side effects)
```

**Balanced:**
```
Write a Python function that validates email addresses.

Requirements:
- Takes a string, returns bool
- Handles edge cases (empty, None, malformed)
- Follow PEP 8 conventions
- Include basic docstring

I'll handle: linting, formatting, detailed type hints
```

### Example 2: Content Generation

**Under-specified:**
```
Write something about our new feature.
```

**Balanced:**
```
Write a feature announcement for our blog.

Feature: AI-powered code review that catches bugs before PR
Audience: Developers already using our product
Tone: Excited but not hyperbolic
Length: 200-300 words

Include:
- Hook explaining the problem this solves
- 2-3 specific capabilities
- How to enable it (Settings > Beta Features)
- Link to documentation: docs.example.com/ai-review
```

### Example 3: Data Processing

**Over-constrained:**
```
Parse this CSV data.
Each row represents one order.
Column 1 is order_id (integer, 6-8 digits).
Column 2 is customer_name (string, max 100 chars).
Column 3 is order_date (ISO 8601 format only).
Column 4 is total (float, 2 decimal places, positive only).
Column 5 is status (must be one of: pending, shipped, delivered).
Skip header row if present.
Handle UTF-8 encoding.
Report any rows that don't match expected format.
Output as JSON array.
Each JSON object must have keys matching column names.
Values must preserve original types.
...
```

**Balanced:**
```
Parse this CSV of orders into JSON.

CSV columns: order_id, customer_name, order_date, total, status
Expected output: Array of objects with these fields
Edge cases: Skip malformed rows, note count of skipped in response
```

The model knows how to parse CSVs. You don't need to explain CSV mechanics.

---

## The "Model Knows" Heuristic

Models have substantial built-in knowledge. You don't need to specify:

| Don't Specify | Model Already Knows |
|---------------|---------------------|
| "Use proper grammar" | Standard for all output |
| "JSON uses braces and brackets" | Syntax of common formats |
| "Python uses indentation" | Language fundamentals |
| "Be helpful" | Base instruction-following behavior |
| "Answer the question asked" | Fundamental task understanding |

**DO specify things the model can't know:**
- Your specific business context
- Non-standard requirements
- Preferences that differ from defaults
- Information from after training cutoff

---

## Hands-on Exercise

### Your Task

You have two prompts—one over-constrained, one under-specified. Fix both.

**Prompt A (Over-constrained):**
```
Write a function to calculate shipping costs. Use Python 3.10+ 
syntax. Use exactly 4-space indentation. Function name must be 
calculate_shipping_cost. First parameter must be weight_kg of 
type float. Second parameter must be distance_km of type float. 
Third parameter must be shipping_method of type str with allowed 
values "standard", "express", "overnight". Return type must be 
float representing cost in USD with exactly 2 decimal places. 
Use match statement for shipping method selection. Standard rate 
is $0.50/kg + $0.10/km. Express is $0.75/kg + $0.15/km. Overnight 
is $1.00/kg + $0.25/km. Add $5 flat fee for express, $15 for 
overnight. Raise ValueError if weight or distance is negative. 
Raise ValueError if shipping_method is not valid. Include 
docstring. Docstring must explain parameters. Docstring must 
explain return value. Docstring must list possible exceptions. 
Maximum function length 30 lines.
```

**Prompt B (Under-specified):**
```
Help me with my code.
```

### Requirements

1. Revise Prompt A to be appropriately constrained (keep essential, remove excessive)
2. Revise Prompt B to be properly specified (add necessary context/requirements)
3. For each, list what you removed or added and why

<details>
<summary>💡 Hints (click to expand)</summary>

**For Prompt A:**
- Which constraints define the *requirement* vs. *style*?
- Can the model figure out the match statement is appropriate?
- Is specifying "exactly 4 spaces" essential or preference?

**For Prompt B:**
- What code? What kind of help?
- What's the current problem?
- What's the expected behavior?
- What has already been tried?

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**Prompt A Revised:**

```
Write a Python function `calculate_shipping_cost(weight_kg, distance_km, method)`.

Pricing:
- Standard: $0.50/kg + $0.10/km
- Express: $0.75/kg + $0.15/km + $5 flat
- Overnight: $1.00/kg + $0.25/km + $15 flat

Return cost as float. Raise ValueError for invalid inputs (negative values, unknown method).

Include a docstring documenting parameters and exceptions.
```

**What was removed and why:**

| Removed | Why |
|---------|-----|
| "Use Python 3.10+ syntax" | Model uses modern syntax by default |
| "4-space indentation" | PEP 8 default, model knows this |
| "type hints for float" | Implicit in the description |
| "use match statement" | Model can choose best construct |
| "exactly 2 decimal places" | Float handling is standard |
| "maximum 30 lines" | Not a real requirement |
| Detailed docstring requirements | "Include docstring" is sufficient |

---

**Prompt B Revised:**

```
Debug this Python function that calculates order totals.

Current behavior: Returns incorrect total when discount > 0
Expected behavior: Apply discount as percentage (e.g., 10 = 10% off)

def calculate_total(items, discount):
    subtotal = sum(item['price'] * item['quantity'] for item in items)
    total = subtotal - discount  # Bug is here
    return total

What I've tried:
- Verified items list is correct
- Discount value is correct (e.g., 10 for 10%)

Please: Identify the bug and provide the fix.
```

**What was added and why:**

| Added | Why |
|-------|-----|
| Language (Python) | Model needs to know language |
| Function purpose | Context for understanding |
| Current vs. expected behavior | Defines the bug |
| Actual code | What to debug |
| What's been tried | Avoids redundant suggestions |
| Specific ask | Clear deliverable |

</details>

---

## Summary

✅ Over-constraining: Too many rules → confused/degraded output
✅ Under-specifying: Too few details → wrong assumptions
✅ Apply "Essential Constraints Only" principle
✅ Start minimal, add constraints only when testing reveals need
✅ Audit existing prompts to remove redundant/cosmetic constraints
✅ Models have built-in knowledge—specify what they can't know

**Next:** [Token Waste](./04-token-waste.md)

---

## Further Reading

- [Anthropic Prompting Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/) - Balancing specificity
- [Google Gemini Constraints](https://ai.google.dev/gemini-api/docs/prompting-strategies#add_constraints) - When to add/remove constraints
- [OpenAI Prompt Guide](https://platform.openai.com/docs/guides/prompt-engineering) - Clear instruction patterns

---

<!-- 
Sources Consulted:
- Anthropic: "Be explicit" but avoid over-constraining
- Google Gemini: Clear and specific without excessive detail
- OpenAI: Balance between guidance and model capability
-->
