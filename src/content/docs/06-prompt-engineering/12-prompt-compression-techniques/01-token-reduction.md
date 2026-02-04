---
title: "Token Reduction Strategies"
---

# Token Reduction Strategies

## Introduction

Every word in your prompt consumes tokens—and tokens cost money and time. Token reduction is the art of saying more with less: conveying the same instructions using fewer tokens without degrading output quality. This lesson teaches practical techniques for writing leaner prompts.

> **🔑 Key Insight:** Models understand terse instructions as well as verbose ones. The extra words are often for human readers, not the model.

### What We'll Cover

- Synonym selection for brevity
- Removing filler words and phrases
- Terse instruction style
- Abbreviation strategies
- Testing compressed prompts

### Prerequisites

- [Prompt Compression Overview](./00-prompt-compression-overview.md)
- Understanding of tokenization basics

---

## Understanding Token Economics

### How Tokens Work

| Text | Approximate Tokens |
|------|-------------------|
| "Hello" | 1 token |
| "Hello, how are you today?" | 6 tokens |
| "Hi" | 1 token |
| Common words | 1 token each |
| Uncommon words | 2-4 tokens |
| Numbers | 1-2 tokens per digit group |

### The 80/20 Rule of Prompts

Most prompts contain significant redundancy:

| Component | Typical Waste |
|-----------|---------------|
| Filler phrases | 15-25% of tokens |
| Redundant instructions | 10-20% of tokens |
| Verbose formatting | 5-15% of tokens |
| Over-explained examples | 10-20% of tokens |

---

## Filler Word Removal

### Common Filler Phrases

| Verbose | Concise | Tokens Saved |
|---------|---------|--------------|
| "Please carefully analyze" | "Analyze" | 2 |
| "I would like you to" | (remove entirely) | 5 |
| "Make sure to" | (remove entirely) | 3 |
| "It is important that you" | (remove entirely) | 5 |
| "In order to" | "To" | 2 |
| "Due to the fact that" | "Because" | 4 |
| "At this point in time" | "Now" | 4 |
| "In the event that" | "If" | 3 |

### Before and After Examples

**Verbose (47 tokens):**
```
Please carefully analyze the following text and provide a comprehensive 
summary that captures all of the key points and main ideas. Make sure 
to include any important details that might be relevant to understanding 
the overall meaning of the content.
```

**Concise (12 tokens):**
```
Summarize the key points and important details:
```

**Token savings:** 74% reduction

### Removing Politeness Tokens

Models don't need social niceties:

```diff
- Please help me with the following task. I would really appreciate 
- it if you could analyze this data.
+ Analyze this data:
```

> **Note:** Politeness doesn't improve output quality, but tone instructions ("be friendly") do affect response style.

---

## Terse Instruction Style

### Imperative Over Descriptive

| Descriptive (verbose) | Imperative (concise) |
|-----------------------|---------------------|
| "You should extract the entities" | "Extract entities" |
| "Your task is to classify" | "Classify" |
| "You will need to summarize" | "Summarize" |
| "I want you to translate" | "Translate" |

### Removing Unnecessary Context

**Verbose:**
```
You are an AI assistant designed to help users with various tasks. 
One of your capabilities is to analyze sentiment in text. When a user 
provides you with text, you should determine whether the sentiment 
is positive, negative, or neutral and explain your reasoning.
```

**Concise:**
```
Analyze sentiment as positive/negative/neutral. Explain reasoning.
```

### Using Symbols and Formatting

Replace words with symbols where clear:

| Verbose | Symbol | Tokens Saved |
|---------|--------|--------------|
| "greater than or equal to" | "≥" or ">=" | 4 |
| "for example" | "e.g." | 1 |
| "and so on" | "etc." | 2 |
| "input arrow output" | "input → output" | 1 |
| "Option 1, Option 2, Option 3" | "Options: 1/2/3" | 2 |

---

## Abbreviation Strategies

### When Abbreviations Work

| Context | Abbreviation Works | Example |
|---------|-------------------|---------|
| Technical domains | ✅ | API, JSON, SQL |
| Common terms | ✅ | info, doc, msg |
| Output field names | ✅ | "cont" for "continuation" |
| User-facing content | ❌ | Keep readable |
| Ambiguous terms | ❌ | "dev" could be developer or development |

### Shortening JSON Keys

JSON field names are generated tokens. Shorten them:

**Verbose (25+ output tokens for keys):**
```json
{
  "message_is_conversation_continuation": true,
  "number_of_messages_so_far": 5,
  "user_sentiment": "frustrated",
  "response_requirements": "apologize and offer solution"
}
```

**Concise (10 output tokens for keys):**
```json
{
  "cont": true,
  "msg_count": 5,
  "sentiment": "frustrated", 
  "reqs": "apologize, offer solution"
}
```

### Defining Abbreviations

When using non-obvious abbreviations, define them once:

```
Abbreviations: cont=continuation, msg=message, usr=user, sys=system

Analyze the usr msg and return:
- cont: bool (is this a follow-up?)
- sentiment: positive/negative/neutral
```

---

## Synonym Selection

### Choosing Shorter Words

| Longer Word | Shorter Synonym | Token Impact |
|-------------|-----------------|--------------|
| approximately | about, ~, roughly | -1 |
| comprehensive | full, complete | -1 |
| subsequently | then, next | -1 |
| utilization | use | -2 |
| implementation | setup, build | -1 |
| functionality | feature | -1 |
| configuration | config, setup | -1 |

### Domain-Specific Jargon

Technical domains have accepted short forms:

| Domain | Long Form | Short Form |
|--------|-----------|------------|
| Programming | function | fn, func |
| Programming | variable | var |
| Programming | parameter | param |
| Data | database | db |
| Web | authentication | auth |
| ML | configuration | config |

---

## Structured Compression Techniques

### Lists Over Prose

**Prose (32 tokens):**
```
When analyzing the text, you should look at the overall sentiment, 
identify any named entities mentioned, extract key topics, and 
determine the language of the text.
```

**List (15 tokens):**
```
Extract:
- sentiment
- named entities  
- key topics
- language
```

### Tables for Multiple Rules

**Prose (45 tokens):**
```
For customer support, use a friendly tone. For technical documentation, 
use a formal tone. For social media, use a casual tone. For legal 
content, use a precise tone.
```

**Table (25 tokens):**
```
Context | Tone
support | friendly
docs | formal
social | casual
legal | precise
```

### Colon Notation

```diff
- The input will be a customer message. The output should be a JSON 
- object containing the following fields.
+ Input: customer message
+ Output: JSON with fields below
```

---

## Testing Compressed Prompts

### Compression Quality Framework

```python
from dataclasses import dataclass

@dataclass
class CompressionTest:
    original_prompt: str
    compressed_prompt: str
    test_cases: list[dict]
    quality_threshold: float = 0.95  # 95% quality retention required

def test_compression(test: CompressionTest, model: str) -> dict:
    """Test if compression maintains quality."""
    
    original_scores = []
    compressed_scores = []
    
    for case in test.test_cases:
        # Run both prompts
        orig_output = call_model(test.original_prompt, case["input"], model)
        comp_output = call_model(test.compressed_prompt, case["input"], model)
        
        # Score outputs
        orig_score = evaluate_output(orig_output, case["expected"])
        comp_score = evaluate_output(comp_output, case["expected"])
        
        original_scores.append(orig_score)
        compressed_scores.append(comp_score)
    
    # Calculate metrics
    orig_avg = sum(original_scores) / len(original_scores)
    comp_avg = sum(compressed_scores) / len(compressed_scores)
    quality_retention = comp_avg / orig_avg if orig_avg > 0 else 0
    
    # Token analysis
    orig_tokens = count_tokens(test.original_prompt)
    comp_tokens = count_tokens(test.compressed_prompt)
    token_reduction = 1 - (comp_tokens / orig_tokens)
    
    return {
        "original_quality": orig_avg,
        "compressed_quality": comp_avg,
        "quality_retention": quality_retention,
        "passes_threshold": quality_retention >= test.quality_threshold,
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "token_reduction": token_reduction,
        "recommendation": generate_recommendation(quality_retention, token_reduction)
    }

def generate_recommendation(quality: float, reduction: float) -> str:
    if quality >= 0.95 and reduction >= 0.30:
        return "EXCELLENT: Significant savings with quality maintained"
    elif quality >= 0.95:
        return "GOOD: Quality maintained, consider more compression"
    elif quality >= 0.90 and reduction >= 0.40:
        return "ACCEPTABLE: Minor quality loss offset by major savings"
    else:
        return "REJECT: Quality degradation too significant"
```

### A/B Testing Compression

```python
def ab_test_compression(
    original: str,
    compressed: str,
    test_cases: list[dict],
    n_runs: int = 100
) -> dict:
    """Run statistically rigorous A/B test."""
    
    results = {"original": [], "compressed": []}
    
    for case in test_cases[:n_runs]:
        # Randomize order to avoid position bias
        prompts = [("original", original), ("compressed", compressed)]
        random.shuffle(prompts)
        
        for name, prompt in prompts:
            output = call_model(prompt, case["input"])
            score = evaluate_output(output, case["expected"])
            results[name].append(score)
    
    # Statistical comparison
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(results["original"], results["compressed"])
    
    orig_mean = sum(results["original"]) / len(results["original"])
    comp_mean = sum(results["compressed"]) / len(results["compressed"])
    
    return {
        "original_mean": orig_mean,
        "compressed_mean": comp_mean,
        "difference": comp_mean - orig_mean,
        "p_value": p_value,
        "significant_difference": p_value < 0.05,
        "recommendation": (
            "Compression safe" if p_value > 0.05 or comp_mean >= orig_mean
            else "Compression degrades quality"
        )
    }
```

---

## Hands-on Exercise

### Your Task

Compress the following verbose prompt while maintaining quality.

**Original Prompt (87 tokens):**
```
You are a helpful AI assistant that specializes in analyzing customer 
feedback. I would like you to please carefully read through the 
following customer review and perform a comprehensive analysis. You 
should identify the overall sentiment of the review, extract any 
specific product features that are mentioned, note any complaints or 
issues raised, and provide a brief summary of the main points. Please 
format your response as a JSON object.
```

### Requirements

1. Reduce to under 30 tokens
2. Maintain all functional requirements
3. Keep the JSON output format instruction

<details>
<summary>💡 Hints (click to expand)</summary>

- Remove "You are a helpful AI assistant that specializes in..."
- Remove "I would like you to please carefully..."
- Use list format for extraction tasks
- Use imperative verbs

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**Compressed Prompt (24 tokens):**
```
Analyze this customer review. Extract:
- sentiment
- product features mentioned
- complaints/issues
- summary

Output: JSON
```

**Alternative (21 tokens):**
```
Review analysis → JSON:
- sentiment
- features mentioned
- complaints
- summary
```

**Token Reduction:** 75% (87 → 24 tokens)

**Verification:** Both prompts produce equivalent output when tested on sample reviews.

</details>

---

## Common Compression Mistakes

| ❌ Mistake | ✅ Better Approach |
|-----------|-------------------|
| Removing critical constraints | Keep safety/format constraints |
| Over-abbreviating user-facing output | Only abbreviate internal fields |
| Removing all examples | Keep 1-2 representative examples |
| Compressing past understanding | Test before deploying |
| One-size-fits-all compression | Task-specific optimization |

---

## Summary

✅ Remove filler phrases: "Please carefully" → direct verbs
✅ Use imperative style: "Extract" not "You should extract"
✅ Choose shorter synonyms: "comprehensive" → "full"
✅ Abbreviate JSON keys: "message_continuation" → "cont"
✅ Use lists over prose for multiple items
✅ Always test compressed prompts for quality retention

**Next:** [Removing Redundancy](./02-redundancy-removal.md)

---

## Further Reading

- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) - Test token counts
- [Anthropic Token Counting](https://docs.anthropic.com/en/docs/build-with-claude/token-counting) - Claude tokenization
- [Latency Optimization Guide](https://platform.openai.com/docs/guides/latency-optimization) - OpenAI best practices

---

<!-- 
Sources Consulted:
- OpenAI Latency Optimization Guide: Generate fewer tokens, use fewer input tokens
- Token counting documentation from OpenAI and Anthropic
- Production prompt optimization patterns
-->
