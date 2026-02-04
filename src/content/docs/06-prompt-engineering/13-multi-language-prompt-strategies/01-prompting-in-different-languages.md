---
title: "Prompting in Different Languages"
---

# Prompting in Different Languages

## Introduction

Should you prompt in English and request output in another language, or prompt entirely in the target language? This fundamental decision affects output quality, cost, and user experience. This lesson explores when each approach works best and how to optimize for different language scenarios.

> **🔑 Key Insight:** For Tier 1-2 languages, native prompting produces more natural outputs. For Tier 4-5 languages, English prompts with translation often yield better results.

### What We'll Cover

- Native language prompting patterns
- English prompts for non-English output
- Model language capability assessment
- Quality variations by language
- Practical decision framework

### Prerequisites

- [Multi-Language Overview](./00-multi-language-overview.md)
- Understanding of prompt structure basics

---

## The Language Prompting Spectrum

### Three Approaches

| Approach | Description | Best For |
|----------|-------------|----------|
| **Fully Native** | Prompt and output in target language | Tier 1-2 languages, cultural content |
| **Hybrid** | English instructions, native examples/output | Tier 3 languages, complex tasks |
| **English + Translate** | English prompt, translate output | Tier 4-5 languages, consistency |

### Fully Native Prompting

```python
# French prompt for French output
prompt_fr = """
Vous êtes un assistant culinaire français spécialisé dans la cuisine traditionnelle.

Tâche: Suggérez un menu de trois plats pour un dîner d'automne.

Contraintes:
- Utilisez des ingrédients de saison
- Incluez une entrée, un plat principal, et un dessert
- Chaque plat doit inclure une brève description

Format: JSON avec les clés "entrée", "plat", "dessert"
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt_fr}]
)
```

**Advantages:**
- Most natural phrasing and idioms
- Cultural authenticity
- No translation artifacts

**Disadvantages:**
- Requires language expertise to write/verify prompts
- May perform worse for complex reasoning in lower-tier languages

### Hybrid Approach

```python
# English instructions with French examples and output
prompt_hybrid = """
You are a French culinary assistant specializing in traditional cuisine.

Task: Suggest a three-course menu for an autumn dinner.

Constraints:
- Use seasonal ingredients
- Include starter, main course, and dessert
- Each dish needs a brief description

Respond entirely in French. Here's an example format:

<exemple>
Entrée: Velouté de potiron - Une soupe onctueuse aux notes de muscade
Plat: Boeuf bourguignon - Un classique mijoté aux légumes d'automne
Dessert: Tarte aux pommes - Croûte feuilletée, pommes caramélisées
</exemple>

Output: JSON with keys "entrée", "plat", "dessert"
"""
```

**Advantages:**
- Clear instructions (model performs best understanding English)
- Native examples teach desired style
- Good balance for Tier 2-3 languages

**Disadvantages:**
- Language mixing can confuse some models
- May produce slightly less natural output

### English + Translation Pipeline

```python
# Step 1: Generate in English
english_prompt = """
You are a culinary expert specializing in French cuisine.

Task: Suggest a three-course autumn dinner menu.

Requirements:
- Seasonal ingredients
- Include appetizer, main course, dessert
- Brief description for each

Output: JSON with keys "appetizer", "main", "dessert"
"""

english_response = generate(english_prompt)

# Step 2: Translate to French
translation_prompt = f"""
Translate this menu to French. Maintain culinary terminology and 
natural French phrasing. Use appropriate French culinary terms.

Content to translate:
{english_response}

Output the same JSON structure with French text.
"""

french_response = generate(translation_prompt)
```

**Advantages:**
- Maximum reasoning quality (in English)
- Consistent translation quality
- Works for any language

**Disadvantages:**
- Two API calls (cost + latency)
- Potential translation artifacts
- Loss of cultural nuance

---

## Language Capability by Model

### Model Strengths by Language Family

| Model | Romance Languages | Germanic | CJK | Arabic | Indic |
|-------|-------------------|----------|-----|--------|-------|
| GPT-4o | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★★ |
| GPT-4o-mini | ★★★★ | ★★★★ | ★★★ | ★★★ | ★★★ |
| Claude 3.5 Sonnet | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★★ |
| Claude 3 Haiku | ★★★★ | ★★★★ | ★★★ | ★★★ | ★★ |
| Gemini 1.5 Pro | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ |

### Testing Model Capability

```python
def test_language_capability(
    model: str,
    language: str,
    test_prompts: list[dict]
) -> dict:
    """Test model capability in a specific language."""
    
    results = {
        "comprehension": [],
        "generation": [],
        "reasoning": [],
        "cultural": []
    }
    
    for test in test_prompts:
        # Native language prompt
        native_response = generate(model, test["native_prompt"])
        
        # English prompt for same task
        english_response = generate(model, test["english_prompt"])
        
        # Score responses
        native_score = evaluate_response(
            native_response, 
            test["expected"],
            language
        )
        english_score = evaluate_response(
            english_response,
            test["expected"],
            language
        )
        
        results[test["category"]].append({
            "native_score": native_score,
            "english_score": english_score,
            "delta": native_score - english_score
        })
    
    # Aggregate results
    return {
        category: {
            "avg_native": sum(r["native_score"] for r in scores) / len(scores),
            "avg_english": sum(r["english_score"] for r in scores) / len(scores),
            "recommendation": (
                "NATIVE" if sum(r["delta"] for r in scores) > 0 
                else "ENGLISH"
            )
        }
        for category, scores in results.items()
    }
```

---

## Quality Variations by Language

### Common Quality Issues

| Issue | Description | Languages Affected |
|-------|-------------|-------------------|
| **Code-switching** | Mixing languages unexpectedly | All non-English |
| **Formality mismatch** | Wrong register (formal/informal) | Japanese, Korean, German |
| **Idiom literal translation** | Translating idioms word-for-word | All |
| **Script errors** | Wrong characters or diacritics | Arabic, CJK, Vietnamese |
| **Grammar mistakes** | Gender, case, conjugation errors | Romance, Slavic, German |

### Quality Assessment Framework

```python
def assess_output_quality(
    output: str,
    language: str,
    criteria: dict
) -> dict:
    """Assess quality of non-English output."""
    
    quality_prompt = f"""
    Evaluate this {language} text for quality. Score each criterion 1-5.
    
    Text: {output}
    
    Criteria:
    1. Grammar correctness (gender, case, conjugation)
    2. Natural phrasing (sounds native, not translated)
    3. Appropriate formality level
    4. Cultural appropriateness
    5. Technical accuracy (if applicable)
    
    Return JSON with scores and specific issues found.
    """
    
    # Use a strong model to evaluate
    evaluation = generate(
        model="gpt-4o",
        prompt=quality_prompt,
        response_format={"type": "json_object"}
    )
    
    return json.loads(evaluation)
```

### Language-Specific Quality Checks

```python
# Japanese formality check
japanese_checks = {
    "keigo": "Contains appropriate honorific language",
    "particles": "Correct particle usage (は, が, を, etc.)",
    "counters": "Appropriate counter words",
    "politeness": "Consistent politeness level throughout"
}

# Arabic quality checks
arabic_checks = {
    "root_accuracy": "Correct root derivations",
    "gender_agreement": "Noun-adjective gender agreement",
    "definite_articles": "Proper use of ال",
    "dialect_consistency": "Consistent MSA or dialect"
}

# German quality checks
german_checks = {
    "case_accuracy": "Correct nominative/accusative/dative/genitive",
    "gender_accuracy": "Correct der/die/das usage",
    "compound_words": "Proper compound word formation",
    "verb_position": "Correct verb placement"
}
```

---

## Prompting Best Practices by Language Tier

### Tier 1 Languages (95-100% of English)

**Languages:** Spanish, French, German, Portuguese, Italian, Indonesian

```python
# Direct native prompting works well
system_prompt_es = """
Eres un experto en atención al cliente para una tienda de tecnología.

Reglas:
- Usa un tono profesional pero amigable
- Responde siempre en español
- Si no sabes algo, dilo honestamente
- Ofrece alternativas cuando sea posible
"""
```

**Recommendations:**
- ✅ Prompt entirely in target language
- ✅ Use native examples
- ✅ Expect high-quality output
- ⚠️ Still verify with native speakers

### Tier 2 Languages (90-95%)

**Languages:** Arabic, Chinese, Japanese, Korean, Hindi

```python
# Hybrid approach often works best
system_prompt = """
You are a customer service expert for a technology store.

CRITICAL: All responses must be in simplified Chinese (简体中文).

Rules:
- Professional but friendly tone
- If uncertain, acknowledge it honestly
- Offer alternatives when possible

Example response style:
<example>
用户: 这款手机电池能用多久？
回复: 这款手机配备4500mAh电池，正常使用可持续一整天。如果您是重度用户，
我们也有配套的快充移动电源可以推荐给您。
</example>
"""
```

**Recommendations:**
- ✅ English instructions for clarity
- ✅ Native examples for style
- ✅ Explicit language enforcement
- ⚠️ Test complex reasoning tasks

### Tier 3-4 Languages (75-90%)

**Languages:** Bengali, Swahili, Thai, Vietnamese

```python
# Consider translation pipeline
def generate_for_tier_3(prompt: str, target_language: str) -> str:
    # Step 1: Generate in English
    english_result = generate(
        model="gpt-4o",
        prompt=prompt,
        system="You are a helpful assistant. Respond in English."
    )
    
    # Step 2: Translate with cultural adaptation
    translation = generate(
        model="gpt-4o",
        prompt=f"""
        Translate to {target_language} with cultural adaptation.
        
        Original: {english_result}
        
        Guidelines:
        - Adapt idioms to local equivalents
        - Use appropriate formality for the culture
        - Maintain the intent and tone
        """
    )
    
    return translation
```

**Recommendations:**
- ✅ Use English for complex reasoning
- ✅ Dedicated translation step
- ✅ Cultural adaptation instructions
- ⚠️ Quality verification essential

### Tier 5 Languages (<75%)

**Languages:** Yoruba, Amharic, less-resourced languages

```python
# Maximum support approach
def generate_for_tier_5(
    task: str, 
    target_language: str,
    reference_materials: list[str]
) -> str:
    # Provide reference materials for the model
    context = "\n".join(reference_materials)
    
    prompt = f"""
    Task: {task}
    Target language: {target_language}
    
    Reference materials in {target_language}:
    {context}
    
    Instructions:
    1. Complete the task in English first
    2. Translate to {target_language}
    3. Use vocabulary and style from reference materials
    4. If uncertain about any terms, provide alternatives
    """
    
    return generate(model="gpt-4o", prompt=prompt)
```

**Recommendations:**
- ✅ Always use English for reasoning
- ✅ Provide reference materials
- ✅ Multiple alternative outputs
- ⚠️ Always verify with native speakers
- ⚠️ Consider hybrid human-AI workflows

---

## Explicit Language Instructions

### The Importance of Clarity

```python
# ❌ Vague - may cause language drift
prompt_vague = "Explain quantum computing"

# ✅ Explicit - consistent output
prompt_explicit = """
Explain quantum computing.

Language: Spanish (España)
Register: Formal
Audience: University students
"""

# ✅✅ Very explicit - maximum control
prompt_very_explicit = """
Explain quantum computing.

LANGUAGE REQUIREMENTS:
- Output language: Spanish (Castilian Spanish from Spain, not Latin American)
- Register: Formal academic
- Audience: University undergraduate students
- Avoid: Anglicisms where Spanish terms exist
- Use: Standard RAE-approved terminology
"""
```

### Preventing Language Drift in Conversations

```python
def create_multilingual_conversation_prompt(target_language: str) -> str:
    """Create a prompt that maintains language consistency."""
    
    return f"""
    You are a helpful assistant.
    
    CRITICAL LANGUAGE RULES:
    1. ALWAYS respond in {target_language}
    2. Even if the user switches to another language, respond in {target_language}
    3. If you must include technical terms, provide the {target_language} equivalent
    4. Never mix languages within a single response
    
    If the user's message is in a different language, you may:
    - Understand it in any language
    - But you MUST respond only in {target_language}
    """
```

---

## Hands-on Exercise

### Your Task

Build a prompt system that generates product descriptions in three languages (English, Spanish, Japanese) with appropriate cultural adaptation.

**Requirements:**
1. Single input (product details in English)
2. Three outputs with cultural adaptation
3. Quality scoring for each output
4. Cost tracking per language

<details>
<summary>💡 Hints (click to expand)</summary>

- Use appropriate prompting strategy per language tier
- Spanish is Tier 1 (direct native)
- Japanese is Tier 2 (hybrid approach)
- Consider cultural differences in marketing style

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from dataclasses import dataclass

@dataclass
class ProductDescription:
    language: str
    text: str
    tokens_used: int
    quality_score: float

def generate_multilingual_descriptions(
    product_name: str,
    features: list[str],
    target_audience: str
) -> dict[str, ProductDescription]:
    """Generate culturally-adapted product descriptions."""
    
    results = {}
    
    # English (baseline)
    english_prompt = f"""
    Write a compelling product description for {product_name}.
    
    Features: {', '.join(features)}
    Target audience: {target_audience}
    
    Style: Engaging, benefits-focused, 100-150 words
    """
    
    english_result = generate_with_tracking(english_prompt)
    results["english"] = ProductDescription(
        language="English",
        text=english_result["text"],
        tokens_used=english_result["tokens"],
        quality_score=1.0  # Baseline
    )
    
    # Spanish (Tier 1 - direct native)
    spanish_prompt = f"""
    Escribe una descripción de producto atractiva para {product_name}.
    
    Características: {', '.join(features)}
    Audiencia objetivo: {target_audience}
    
    Estilo: 
    - Atractivo y enfocado en beneficios
    - 100-150 palabras
    - Tono cercano pero profesional
    - Adapta el enfoque de marketing para el mercado hispanohablante
    """
    
    spanish_result = generate_with_tracking(spanish_prompt)
    spanish_quality = assess_output_quality(spanish_result["text"], "Spanish")
    
    results["spanish"] = ProductDescription(
        language="Spanish",
        text=spanish_result["text"],
        tokens_used=spanish_result["tokens"],
        quality_score=spanish_quality["overall"]
    )
    
    # Japanese (Tier 2 - hybrid approach)
    japanese_prompt = f"""
    Write a product description for {product_name} in Japanese.
    
    Features: {', '.join(features)}
    Target audience: {target_audience}
    
    REQUIREMENTS:
    - Output entirely in Japanese (日本語)
    - Use polite form (です/ます)
    - Length: 100-150 words equivalent
    - Marketing style: Emphasis on quality and reliability (Japanese consumers value these)
    - Include specific measurements/numbers (Japanese audiences appreciate precision)
    
    Example style:
    <example>
    【商品特徴】
    本製品は、最高品質の素材を使用し、職人の技術で丁寧に作られています。
    </example>
    """
    
    japanese_result = generate_with_tracking(japanese_prompt)
    japanese_quality = assess_output_quality(japanese_result["text"], "Japanese")
    
    results["japanese"] = ProductDescription(
        language="Japanese",
        text=japanese_result["text"],
        tokens_used=japanese_result["tokens"],
        quality_score=japanese_quality["overall"]
    )
    
    return results

# Usage
descriptions = generate_multilingual_descriptions(
    product_name="Wireless Earbuds Pro",
    features=["40-hour battery", "Active noise cancellation", "Water resistant"],
    target_audience="Young professionals"
)

# Display results with cost comparison
for lang, desc in descriptions.items():
    print(f"\n=== {lang.upper()} ===")
    print(f"Quality Score: {desc.quality_score:.2f}")
    print(f"Tokens Used: {desc.tokens_used}")
    print(f"Text:\n{desc.text[:200]}...")
```

</details>

---

## Summary

✅ **Tier 1-2 languages:** Use native or hybrid prompting for best quality
✅ **Tier 3-4 languages:** Consider English + translation pipelines
✅ **Tier 5 languages:** Always use English reasoning with careful translation
✅ **Always be explicit:** Specify language, register, and cultural context
✅ **Test and verify:** Quality varies—always validate with native speakers
✅ **Track costs:** Non-Latin scripts use more tokens

**Next:** [Cross-Lingual Engineering](./02-cross-lingual-engineering.md)

---

## Further Reading

- [Anthropic Multilingual Support](https://docs.anthropic.com/en/docs/build-with-claude/multilingual-support) - Performance data
- [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering) - General best practices
- [Google Gemini Prompting](https://ai.google.dev/gemini-api/docs/prompting-strategies) - Multi-language tips

---

<!-- 
Sources Consulted:
- Anthropic Multilingual Support: Claude performance benchmarks by language
- OpenAI Prompt Engineering Guide: Message formatting, examples
- Google Gemini Prompting Strategies: Input prefixes for language
-->
