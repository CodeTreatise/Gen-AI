---
title: "Token Waste"
---

# Token Waste

## Introduction

Every token costs money and consumes context window space. Wasted tokens don't just increase costs—they can push important information out of the model's attention window, degrading response quality. This lesson teaches you to identify and eliminate token waste while maintaining prompt effectiveness.

> **🔑 Key Insight:** A prompt that's 50% shorter and equally effective is doubly better: lower cost AND more room for useful context.

### What We'll Cover

- Common sources of token waste
- Compression techniques that preserve meaning
- Cost-aware prompt optimization
- Measuring and benchmarking token efficiency

### Prerequisites

- [Common Pitfalls Overview](./00-common-pitfalls-overview.md)
- [Over/Under-Constraining](./03-over-under-constraining.md)
- Understanding of token-based API pricing

---

## Understanding Token Economics

### Why Token Efficiency Matters

```
┌──────────────────────────────────────────────────────────────┐
│               TOKEN IMPACT ZONES                              │
└──────────────────────────────────────────────────────────────┘

                    COST           CONTEXT           ATTENTION
                    ────           ───────           ─────────
                    $ per          Window            Model focus
                    token          limits            distribution

Bloated prompt →    Higher         Fills faster      Dilutes
                    bills          → truncation      important parts

Lean prompt →       Lower          Room for more     Focus on
                    bills          context           what matters
```

### The Numbers

| Model | Input Cost | Output Cost | Context Limit |
|-------|-----------|-------------|---------------|
| GPT-4o | $2.50/1M | $10/1M | 128K |
| GPT-4o-mini | $0.15/1M | $0.60/1M | 128K |
| Claude 3.5 Sonnet | $3/1M | $15/1M | 200K |
| Claude 3 Haiku | $0.25/1M | $1.25/1M | 200K |
| Gemini Pro | $0.075/1M | $0.30/1M | 2M |

**Example calculation:**

A system prompt of 1,000 tokens vs 500 tokens:
- 100,000 API calls/month
- At $2.50/1M tokens
- Savings: 500 × 100,000 = 50M tokens saved = **$125/month**

For high-volume applications, token efficiency directly impacts profitability.

---

## Common Sources of Token Waste

### Source 1: Redundant Instructions

The same instruction stated multiple ways:

❌ **Wasteful (89 tokens):**
```
Be concise in your responses. Keep your answers brief and to the point.
Don't be verbose or long-winded. Avoid unnecessary elaboration. 
Get straight to the point. Don't pad your responses with filler.
Brevity is important.
```

✅ **Efficient (8 tokens):**
```
Keep responses under 100 words.
```

### Source 2: Unnecessary Context

Including information the model doesn't need:

❌ **Wasteful (67 tokens):**
```
I've been working on this project for about 3 months now. It started
as a side project but has grown into something bigger. My team consists
of 4 developers, and we're using an agile methodology with 2-week sprints.
Anyway, I need you to review this code.
```

✅ **Efficient (7 tokens):**
```
Review this code for bugs and performance issues.
```

> **Rule:** If removing it doesn't change the output, remove it.

### Source 3: Verbose Examples

Too many examples, or examples longer than necessary:

❌ **Wasteful (~200 tokens):**
```
Example 1:
User: What is the capital of France?
Assistant: The capital of France is Paris. Paris is a beautiful city known 
for its rich history, culture, and landmarks like the Eiffel Tower. It has
been the capital since the 10th century and is home to over 2 million people
in the city proper, with the metropolitan area housing over 12 million...

Example 2:
User: What is the capital of Germany?
Assistant: The capital of Germany is Berlin. Berlin is the largest city in 
Germany by population and has served as the capital since reunification in 
1990. The city is known for its historical significance, including the Berlin
Wall which divided the city during the Cold War...

Example 3:
User: What is the capital of Italy?
Assistant: The capital of Italy is Rome. Rome is one of the oldest continuously
occupied cities in Europe, founded in 753 BC according to legend. It served
as the capital of the Roman Empire and later became the center of...
```

✅ **Efficient (~40 tokens):**
```
Examples:
Q: What is the capital of France?
A: The capital of France is Paris.

Q: What is the capital of Germany?
A: The capital of Germany is Berlin.
```

**Fewer examples, shorter format, same pattern demonstrated.**

### Source 4: Over-Explanation

Explaining things the model already knows:

❌ **Wasteful (98 tokens):**
```
JSON (JavaScript Object Notation) is a lightweight data-interchange format
that is easy for humans to read and write and easy for machines to parse
and generate. It uses key-value pairs and arrays. Keys must be strings 
enclosed in double quotes. Values can be strings, numbers, booleans, null,
arrays, or objects. Please format your response as JSON.
```

✅ **Efficient (4 tokens):**
```
Respond in JSON format.
```

### Source 5: Repeated System Context

Information repeated across every message:

❌ **Wasteful pattern:**
```
Message 1: "You are a helpful assistant. You work for Acme Corp. You are 
professional and friendly. Be accurate. [actual request]"

Message 2: "You are a helpful assistant. You work for Acme Corp. You are 
professional and friendly. Be accurate. [actual request]"

Message 3: "You are a helpful assistant. You work for Acme Corp. You are 
professional and friendly. Be accurate. [actual request]"
```

✅ **Efficient pattern:**
Put system context in system/developer message (sent once), not repeated in each user message.

---

## Compression Techniques

### Technique 1: Semantic Compression

Reduce words while preserving meaning:

| Original (17 tokens) | Compressed (6 tokens) |
|---------------------|----------------------|
| "Please analyze the following piece of code and identify any potential issues that might cause problems or bugs" | "Analyze this code for bugs:" |

**Compression strategies:**

| Strategy | Before | After |
|----------|--------|-------|
| Remove hedging | "I think maybe you could try..." | "Try..." |
| Remove politeness (in prompts) | "Would you please kindly..." | Direct instruction |
| Use imperatives | "You should analyze..." | "Analyze..." |
| Remove filler | "In order to, for the purpose of" | "To" |
| Combine instructions | "Do A. Then do B. After that, do C." | "Do A, B, then C." |

### Technique 2: Example Minimization

Use the minimum examples needed:

```
# Few-shot efficiency ladder

ZERO-SHOT (0 examples) - Try this first
"Classify sentiment: positive, negative, or neutral."

ONE-SHOT (1 example) - If zero-shot isn't consistent
"Classify sentiment.
Example: 'I love it!' → positive"

FEW-SHOT (2-3 examples) - Only if pattern is complex
"Classify sentiment.
'I love it!' → positive
'I hate it!' → negative
'It's fine.' → neutral"
```

> **Rule of thumb:** Start with zero-shot. Add examples only if outputs are inconsistent or wrong.

### Technique 3: Structured Compression

Use structured formats that are token-efficient:

❌ **Prose (31 tokens):**
```
The product has the following attributes: The name is Widget Pro, 
the price is twenty-five dollars and ninety-nine cents, the category
is Electronics, and it is currently in stock.
```

✅ **Structured (17 tokens):**
```
Product:
- name: Widget Pro
- price: $25.99
- category: Electronics
- in_stock: true
```

✅ **JSON (16 tokens):**
```json
{"name":"Widget Pro","price":25.99,"category":"Electronics","in_stock":true}
```

### Technique 4: Reference Compression

Point to information rather than repeating it:

❌ **Repeated (high tokens):**
```
For each of the following 50 items, analyze the item name, 
determine the item category, calculate the item price with tax,
and format the item output as JSON:

[full context for item 1]
[full context for item 2]
... (50 items with repeated instruction context)
```

✅ **Referenced (lower tokens):**
```
Analyze each item below. For each:
1. Determine category
2. Calculate price with 8% tax
3. Output as JSON

Items:
[item 1]
[item 2]
...
```

### Technique 5: Prompt Caching Optimization

For APIs that support prompt caching, front-load static content:

```python
# OpenAI prompt caching (enabled by default for long prompts)
# Static content is cached if it appears at the START of the prompt

messages = [
    {
        "role": "developer",
        "content": """
        [STATIC CONTENT - will be cached]
        Your detailed system prompt that doesn't change...
        Your examples that don't change...
        Your formatting rules that don't change...
        """
    },
    {
        "role": "user", 
        "content": f"""
        [DYNAMIC CONTENT - changes per request]
        {user_specific_request}
        """
    }
]
```

**Cost impact:**
- Cached tokens: 50% cost reduction (OpenAI)
- Non-cached: Full price

---

## Cost-Aware Prompt Optimization

### Token Budgeting

Set explicit budgets for prompt components:

```
TOTAL TOKEN BUDGET: 2,000 tokens

├── System prompt: 400 tokens (20%)
├── Examples: 200 tokens (10%)
├── Context/data: 1,000 tokens (50%)
└── User request: 400 tokens (20%)
```

### Cost Optimization Workflow

```python
def optimize_prompt_cost(prompt: str, target_tokens: int) -> str:
    """
    Iteratively reduce prompt size while maintaining quality.
    """
    current = tokenize(prompt)
    
    while len(current) > target_tokens:
        # Try compression strategies in order of impact
        
        # 1. Remove redundant instructions
        prompt = deduplicate_instructions(prompt)
        
        # 2. Compress verbose sections
        prompt = compress_verbose_sections(prompt)
        
        # 3. Reduce examples
        prompt = minimize_examples(prompt)
        
        # 4. Remove unnecessary context
        prompt = trim_context(prompt)
        
        # Validate quality still acceptable
        if not quality_check(prompt):
            revert_last_change()
            break
            
        current = tokenize(prompt)
    
    return prompt
```

### Measuring Token Efficiency

Track these metrics:

| Metric | Formula | Target |
|--------|---------|--------|
| Token efficiency ratio | useful_output_tokens / input_tokens | > 1.0 |
| Instruction density | task_instructions / total_instructions | > 0.8 |
| Example efficiency | patterns_learned / example_tokens | Maximize |
| Context utilization | used_context / provided_context | > 0.9 |

---

## Real-World Optimization Example

### Before: Bloated Customer Service Prompt

```
You are a helpful customer service representative for TechCorp, a 
leading technology company that has been in business since 1985. 
We pride ourselves on excellent customer service and technical support.
Our company values include integrity, innovation, and customer satisfaction.

When responding to customers, please be polite and professional at all 
times. Always greet the customer warmly and thank them for reaching out.
Be empathetic to their situation and show that you understand their 
concerns. Try to solve their problem efficiently while maintaining a 
friendly tone.

If you cannot solve the problem yourself, please escalate to a human 
agent. Make sure to explain to the customer what is happening and why 
they need to be transferred. Provide any relevant case details.

Please respond in a structured format with clear sections. Use bullet 
points when listing multiple items. Keep your responses concise but 
thorough. Include all relevant information but avoid unnecessary filler.

Here is an example of a good response:

Customer: "My order hasn't arrived yet and it's been 2 weeks."
Agent: "Hello and thank you for reaching out to TechCorp support! I'm
so sorry to hear that your order hasn't arrived yet. That must be very
frustrating, especially after waiting 2 weeks. I completely understand 
your concern.

Let me look into this right away for you. I can see your order #12345
was shipped on March 1st via standard shipping. According to the tracking
information, it appears the package may have been delayed at a distribution
center.

Here's what I can do to help:
• I'll file a trace request with our shipping partner
• You'll receive an update within 24-48 hours
• If we can't locate the package, I'll arrange a replacement

Is there anything else I can help you with today? We really appreciate 
your patience and your business with TechCorp!"
```

**Token count: ~380 tokens**

### After: Optimized Prompt

```
You are a TechCorp customer service agent.

Behavior:
- Empathetic, professional, solution-focused
- Escalate if unable to resolve
- Structured responses with bullet points

Example:
"My order hasn't arrived (2 weeks)."
→ Apologize → Check order status → Offer specific resolution steps → Confirm satisfaction
```

**Token count: ~70 tokens** (82% reduction)

### Quality Comparison

| Metric | Bloated | Optimized |
|--------|---------|-----------|
| Token count | 380 | 70 |
| Key instructions captured | ✓ | ✓ |
| Behavior defined | ✓ | ✓ |
| Example pattern shown | ✓ | ✓ |
| Output quality | Good | Same |
| Monthly cost (100K calls) | $0.95 | $0.18 |

---

## Token Waste Audit Checklist

Use this checklist to audit existing prompts:

### Red Flags to Look For

| Pattern | Example | Fix |
|---------|---------|-----|
| Repeated synonyms | "brief, short, concise, succinct" | Use one word |
| Unnecessary preamble | "I'd like you to, if you could..." | Direct instruction |
| Over-explained basics | "JSON uses {...}" | Assume model knows |
| Excessive examples | 5+ examples for simple pattern | Reduce to 1-2 |
| Repeated system context | Same intro in every message | Move to system message |
| Narrative when list works | Paragraph listing options | Bullet list |
| Verbose formatting rules | Long prose about format | Show example |

### Quick Compression Checklist

- [ ] Can any instruction be said in fewer words?
- [ ] Are there repeated or synonymous instructions?
- [ ] Can examples be shorter or fewer?
- [ ] Is context included that doesn't affect output?
- [ ] Is anything explained that the model already knows?
- [ ] Can structured format replace prose?
- [ ] Is static content front-loaded for caching?

---

## Hands-on Exercise

### Your Task

Compress this prompt while maintaining its effectiveness:

**Original prompt (approx. 250 tokens):**
```
I would like you to help me write professional email responses. When 
writing these email responses, please make sure to maintain a professional
tone throughout the entire email. The emails should be polite and courteous
while also being direct and getting to the point efficiently.

Please structure the email with a proper greeting at the beginning, then
the main body of the response, and finally a professional sign-off at the
end. Make sure to address any questions or concerns that were raised in
the original email that you are responding to.

If there are multiple points to address, please use bullet points or 
numbered lists to organize the response clearly. This helps the recipient
understand each point separately.

The emails should be concise - ideally no more than 150 words unless the
situation requires more detail. Don't include unnecessary filler or 
padding just to make the email seem longer.

Here is an example of a good response:

Original email: "Can you send me the Q3 report? Also, when is our next meeting?"
Response: "Hi [Name],

Thank you for reaching out. Here are the items you requested:

1. Q3 Report: I've attached the Q3 report to this email. Please let me know
   if you need any specific sections highlighted.
2. Next Meeting: Our next team meeting is scheduled for Thursday, March 15th
   at 2:00 PM in Conference Room B.

Please don't hesitate to reach out if you have any other questions.

Best regards,
[Your name]"
```

### Requirements

1. Reduce to under 80 tokens
2. Preserve all essential instructions
3. Maintain example (compressed)
4. Test that a model following the compressed prompt produces similar quality

<details>
<summary>💡 Hints (click to expand)</summary>

- "Professional" covers polite, courteous, direct
- Format can be shown, not described
- "No unnecessary filler" is redundant with "concise"
- Example can be much shorter while showing same pattern

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**Compressed prompt (~55 tokens):**

```
Write professional email responses.

Format: Greeting → Address each point (use bullets if multiple) → Sign-off
Length: Under 150 words

Example:
Input: "Send Q3 report? When's next meeting?"
Output: 
"Hi [Name],
1. Q3 report attached
2. Next meeting: March 15, 2pm, Room B
Let me know if you need anything else.
Best, [Name]"
```

**What was removed:**

| Removed | Why |
|---------|-----|
| "I would like you to" | Filler |
| Explanation of professional | Model knows what professional means |
| Detailed structure explanation | Example shows structure |
| "Don't include filler" | Redundant with "concise" |
| Long example response | Shortened to essential pattern |
| "Please don't hesitate to reach out" | Not essential to pattern |

**Token savings: ~195 tokens (78% reduction)**

</details>

---

## Summary

✅ Every wasted token costs money and consumes context space
✅ Common waste: redundancy, over-explanation, verbose examples
✅ Compression techniques: semantic compression, example minimization, structured formats
✅ Front-load static content for prompt caching benefits
✅ Measure token efficiency: input/output ratio, instruction density
✅ Audit prompts regularly for compression opportunities

**Next:** [Hallucination Triggers](./05-hallucination-triggers.md)

---

## Further Reading

- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) - Caching mechanics
- [Anthropic Token Counting](https://docs.anthropic.com/en/docs/build-with-claude/token-counting) - Understanding token usage
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) - Interactive token counting tool

---

<!-- 
Sources Consulted:
- OpenAI Prompt Caching documentation
- OpenAI Prompt Engineering: Efficiency recommendations
- Model pricing pages for cost calculations
-->
