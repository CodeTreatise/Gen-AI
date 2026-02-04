---
title: "Handling Multilingual Outputs"
---

# Handling Multilingual Outputs

## Introduction

You've sent a prompt in Japanese, but the model responds in English. Or worse, it mixes languages mid-response. Ensuring consistent, correct language in LLM outputs requires explicit strategies—language detection, enforcement mechanisms, and graceful handling of inevitable edge cases.

> **🔑 Key Insight:** Models don't inherently "know" what language to respond in. They predict the most likely next tokens, which can drift between languages without explicit constraints.

### What We'll Cover

- Language detection and verification
- Enforcing output language consistency
- Handling code-switching (language mixing)
- Managing mixed-language content requirements
- Testing and validation strategies

### Prerequisites

- [Translation in Prompt Pipelines](./04-translation-pipelines.md)
- Understanding of API response handling

---

## Language Detection

### Detecting Output Language

```python
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class LanguageDetectionResult:
    primary_language: str
    confidence: float
    languages_detected: dict[str, float]  # language -> percentage
    is_mixed: bool
    mixed_segments: list[dict] | None

class OutputLanguageDetector:
    """Detect and analyze language in LLM outputs."""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect the language(s) in output text."""
        
        prompt = f"""
Analyze the language(s) used in this text.

TEXT:
{text[:2000]}

OUTPUT FORMAT (JSON):
{{
  "primary_language": "language_name",
  "confidence": 0.0-1.0,
  "languages_detected": {{
    "language1": 0.7,
    "language2": 0.3
  }},
  "is_mixed": true/false,
  "mixed_segments": [
    {{"text": "segment", "language": "lang", "position": "start/middle/end"}}
  ] or null
}}
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return LanguageDetectionResult(
            primary_language=result["primary_language"],
            confidence=result["confidence"],
            languages_detected=result["languages_detected"],
            is_mixed=result["is_mixed"],
            mixed_segments=result.get("mixed_segments")
        )
    
    def verify_expected_language(
        self,
        text: str,
        expected_language: str
    ) -> dict:
        """Verify output matches expected language."""
        
        detection = self.detect_language(text)
        
        matches = detection.primary_language.lower() == expected_language.lower()
        
        return {
            "matches_expected": matches,
            "expected": expected_language,
            "detected": detection.primary_language,
            "confidence": detection.confidence,
            "is_mixed": detection.is_mixed,
            "action_needed": not matches or detection.is_mixed
        }
```

### Fast Language Detection

For high-volume applications, use efficient heuristics:

```python
import re
from collections import Counter

class FastLanguageDetector:
    """Lightweight language detection without API calls."""
    
    # Character range patterns
    PATTERNS = {
        "japanese": {
            "hiragana": r'[\u3040-\u309F]',
            "katakana": r'[\u30A0-\u30FF]',
            "kanji": r'[\u4E00-\u9FFF]'
        },
        "chinese": {
            "han": r'[\u4E00-\u9FFF]'
        },
        "korean": {
            "hangul": r'[\uAC00-\uD7AF]',
            "jamo": r'[\u1100-\u11FF]'
        },
        "arabic": {
            "arabic": r'[\u0600-\u06FF]'
        },
        "hebrew": {
            "hebrew": r'[\u0590-\u05FF]'
        },
        "thai": {
            "thai": r'[\u0E00-\u0E7F]'
        },
        "cyrillic": {
            "cyrillic": r'[\u0400-\u04FF]'
        },
        "greek": {
            "greek": r'[\u0370-\u03FF]'
        },
        "devanagari": {
            "devanagari": r'[\u0900-\u097F]'
        }
    }
    
    def detect(self, text: str) -> dict:
        """Fast character-based language detection."""
        
        char_counts = Counter()
        total_chars = 0
        
        for script, patterns in self.PATTERNS.items():
            for name, pattern in patterns.items():
                count = len(re.findall(pattern, text))
                if count > 0:
                    char_counts[script] += count
                    total_chars += count
        
        # Count Latin characters
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        if latin_count > 0:
            char_counts["latin"] = latin_count
            total_chars += latin_count
        
        if total_chars == 0:
            return {"language": "unknown", "confidence": 0}
        
        # Calculate percentages
        percentages = {
            lang: count / total_chars 
            for lang, count in char_counts.items()
        }
        
        primary = max(percentages, key=percentages.get)
        
        return {
            "language": self._script_to_language(primary),
            "confidence": percentages[primary],
            "script_distribution": percentages,
            "is_mixed": len([p for p in percentages.values() if p > 0.1]) > 1
        }
    
    def _script_to_language(self, script: str) -> str:
        """Map script to likely language."""
        
        script_language_map = {
            "latin": "english",  # Default, could be many languages
            "japanese": "japanese",
            "chinese": "chinese",
            "korean": "korean",
            "arabic": "arabic",
            "hebrew": "hebrew",
            "thai": "thai",
            "cyrillic": "russian",  # Default, could be other Slavic
            "greek": "greek",
            "devanagari": "hindi"  # Default, could be Sanskrit, Marathi, etc.
        }
        
        return script_language_map.get(script, script)

# Usage
detector = FastLanguageDetector()
print(detector.detect("こんにちは、世界！"))
# {'language': 'japanese', 'confidence': 1.0, ...}

print(detector.detect("Hello, 世界!"))
# {'language': 'chinese', 'confidence': 0.5, 'is_mixed': True, ...}
```

---

## Enforcing Language Consistency

### Explicit Language Instructions

```python
def create_language_enforced_prompt(
    task: str,
    user_input: str,
    output_language: str,
    strictness: str = "strict"
) -> str:
    """Create a prompt with explicit language enforcement."""
    
    # Language enforcement levels
    enforcement_text = {
        "strict": f"""
CRITICAL REQUIREMENT: Respond ONLY in {output_language}.
- Every word must be in {output_language}
- Do not include any English or other language
- Technical terms should be in {output_language} or transliterated
- If you cannot express something in {output_language}, describe it
""",
        "moderate": f"""
LANGUAGE REQUIREMENT: Respond primarily in {output_language}.
- Main content must be in {output_language}
- Technical terms and proper nouns may remain in original language
- Code, URLs, and identifiers may be in English
""",
        "flexible": f"""
LANGUAGE PREFERENCE: Prefer {output_language} for the response.
- Use {output_language} for explanations and narrative
- Technical content may use appropriate language
- Match the user's language when appropriate
"""
    }
    
    return f"""
{enforcement_text.get(strictness, enforcement_text["moderate"])}

# Task
{task}

# User Input
{user_input}

# Output
[Respond in {output_language}]
"""
```

### System Message Reinforcement

```python
def create_language_locked_messages(
    language: str,
    task: str,
    user_input: str
) -> list[dict]:
    """Create message sequence with language lock."""
    
    return [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that ONLY responds in {language}.

ABSOLUTE RULES:
1. All responses must be in {language}
2. Never switch to English or any other language
3. If the user writes in another language, still respond in {language}
4. Translate technical terms when possible
5. Use {language} punctuation and formatting conventions

Your task: {task}"""
        },
        {
            "role": "user",
            "content": user_input
        },
        {
            "role": "assistant",
            "content": f"[Responding in {language}]\n\n"  # Prefill to anchor language
        }
    ]
```

### Post-Generation Validation

```python
class LanguageConsistencyValidator:
    """Validate and enforce language consistency in outputs."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.detector = FastLanguageDetector()
    
    def validate_and_correct(
        self,
        output: str,
        expected_language: str,
        max_retries: int = 2
    ) -> dict:
        """Validate output language and correct if needed."""
        
        detection = self.detector.detect(output)
        
        # Check if correction needed
        if detection["language"] == expected_language and not detection["is_mixed"]:
            return {
                "output": output,
                "corrected": False,
                "attempts": 1
            }
        
        # Attempt correction
        for attempt in range(max_retries):
            corrected = self._correct_language(output, expected_language)
            new_detection = self.detector.detect(corrected)
            
            if new_detection["language"] == expected_language and not new_detection["is_mixed"]:
                return {
                    "output": corrected,
                    "corrected": True,
                    "attempts": attempt + 2,
                    "original_language": detection["language"]
                }
        
        # Return best attempt
        return {
            "output": corrected,
            "corrected": True,
            "attempts": max_retries + 1,
            "warning": "Could not fully correct language",
            "detected_language": new_detection["language"]
        }
    
    def _correct_language(self, text: str, target_language: str) -> str:
        """Translate/correct text to target language."""
        
        prompt = f"""
The following text should be entirely in {target_language}, but contains 
content in other languages. Rewrite it completely in {target_language}.

RULES:
- Translate ALL non-{target_language} content
- Maintain the meaning and structure
- Use natural {target_language} phrasing
- Keep proper nouns but transliterate if needed

TEXT:
{text}

OUTPUT: Rewritten text in {target_language} only.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
```

---

## Handling Code-Switching

### What is Code-Switching?

Code-switching occurs when a speaker or text alternates between languages. In LLM outputs, this can be:

| Type | Example | Common Cause |
|------|---------|--------------|
| **Inter-sentential** | "That's great. それでは始めましょう。" | Topic shift |
| **Intra-sentential** | "We need to 確認する the results." | Missing vocabulary |
| **Tag-switching** | "The project is complete, 对吧?" | Discourse markers |

### Code-Switching Detection

```python
@dataclass
class CodeSwitchEvent:
    position: int
    from_language: str
    to_language: str
    text_segment: str
    switch_type: str  # "inter", "intra", "tag"

class CodeSwitchDetector:
    """Detect and analyze code-switching in text."""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def detect_switches(self, text: str) -> list[CodeSwitchEvent]:
        """Detect all code-switching events in text."""
        
        prompt = f"""
Analyze this text for code-switching (mixing languages).

TEXT:
{text}

For each switch, identify:
1. Position (character index)
2. Languages involved
3. The text segment where switch occurs
4. Type: "inter" (between sentences), "intra" (within sentence), "tag" (markers)

OUTPUT FORMAT (JSON):
{{
  "has_code_switching": true/false,
  "switches": [
    {{
      "position": 45,
      "from_language": "english",
      "to_language": "japanese",
      "text_segment": "...the meeting. それでは...",
      "switch_type": "inter"
    }}
  ]
}}
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return [
            CodeSwitchEvent(**switch)
            for switch in result.get("switches", [])
        ]
```

### Code-Switching Strategies

```python
class CodeSwitchHandler:
    """Handle code-switching in outputs based on requirements."""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def handle(
        self,
        text: str,
        primary_language: str,
        strategy: str = "unify"
    ) -> str:
        """Handle code-switching based on strategy."""
        
        strategies = {
            "unify": self._unify_to_primary,
            "preserve": self._preserve_intentional,
            "segment": self._segment_by_language,
            "annotate": self._annotate_switches
        }
        
        handler = strategies.get(strategy, self._unify_to_primary)
        return handler(text, primary_language)
    
    def _unify_to_primary(self, text: str, primary_language: str) -> str:
        """Convert all text to primary language."""
        
        prompt = f"""
Rewrite this text entirely in {primary_language}.
Translate any content in other languages.
Maintain meaning and flow.

TEXT: {text}

OUTPUT: Text in {primary_language} only.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def _preserve_intentional(self, text: str, primary_language: str) -> str:
        """Preserve intentional code-switching, fix accidental."""
        
        prompt = f"""
Analyze this text for code-switching (language mixing).

Keep code-switching that appears INTENTIONAL:
- Proper nouns and names
- Technical terms with no good translation
- Cultural expressions/idioms
- Quoted speech in original language

Fix code-switching that appears ACCIDENTAL:
- Mid-sentence language drift
- Inconsistent terminology
- Output errors

TEXT: {text}
PRIMARY LANGUAGE: {primary_language}

OUTPUT: Cleaned text with intentional switches preserved.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def _segment_by_language(self, text: str, primary_language: str) -> str:
        """Segment text with clear language markers."""
        
        prompt = f"""
Segment this text by language with clear markers.

INPUT: {text}

OUTPUT FORMAT:
[{primary_language.upper()}]
Content in {primary_language}...

[OTHER_LANGUAGE]
Content in other language...

Clearly separate each language section.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def _annotate_switches(self, text: str, primary_language: str) -> str:
        """Annotate code-switches for transparency."""
        
        prompt = f"""
Add inline annotations for any non-{primary_language} content.

INPUT: {text}

FORMAT for foreign content: 
original_text [translation in {primary_language}]

Example:
"The kaizen [continuous improvement] approach..."

OUTPUT: Annotated text.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
```

---

## Managing Mixed-Language Requirements

### When Multiple Languages Are Needed

```python
@dataclass
class LanguageZone:
    name: str
    language: str
    start_marker: str
    end_marker: str

class MultiLanguageOutputManager:
    """Manage outputs requiring multiple languages."""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def create_multilingual_prompt(
        self,
        task: str,
        language_zones: list[LanguageZone]
    ) -> str:
        """Create prompt for controlled multilingual output."""
        
        zone_specs = "\n".join([
            f"- {zone.name}: Write in {zone.language} between "
            f"{zone.start_marker} and {zone.end_marker}"
            for zone in language_zones
        ])
        
        zone_template = "\n\n".join([
            f"{zone.start_marker}\n[{zone.name} in {zone.language}]\n{zone.end_marker}"
            for zone in language_zones
        ])
        
        return f"""
{task}

LANGUAGE ZONES:
{zone_specs}

OUTPUT TEMPLATE:
{zone_template}

RULES:
- Each zone must use ONLY its specified language
- Do not mix languages within zones
- Maintain consistent meaning across translations
- Follow natural conventions for each language
"""
    
    def generate_parallel_content(
        self,
        content: str,
        languages: list[str],
        format: str = "side_by_side"
    ) -> dict:
        """Generate content in multiple languages."""
        
        if format == "side_by_side":
            return self._generate_side_by_side(content, languages)
        elif format == "sequential":
            return self._generate_sequential(content, languages)
        else:
            return self._generate_structured(content, languages)
    
    def _generate_structured(
        self,
        content: str,
        languages: list[str]
    ) -> dict:
        """Generate structured multilingual output."""
        
        lang_list = ", ".join(languages)
        
        prompt = f"""
Translate/adapt this content to each language.

CONTENT:
{content}

LANGUAGES NEEDED: {lang_list}

OUTPUT FORMAT (JSON):
{{
  "english": "English content here",
  "japanese": "日本語の内容",
  "spanish": "Contenido en español"
}}

REQUIREMENTS:
- Each version should feel native, not translated
- Adapt cultural references appropriately
- Maintain consistent meaning
- Use appropriate formality for each language
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
```

### Bilingual Output Patterns

```python
class BilingualOutputGenerator:
    """Generate outputs designed for bilingual contexts."""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def generate_with_translations(
        self,
        content: str,
        primary_language: str,
        translation_language: str,
        format: str = "inline"
    ) -> str:
        """Generate content with inline or appended translations."""
        
        formats = {
            "inline": f"""
Create {primary_language} content with {translation_language} translations inline.

FORMAT: Main text (translation in parentheses)
EXAMPLE: "お疲れ様です (Thank you for your hard work)"

CONTENT TO PROCESS:
{content}
""",
            "footnote": f"""
Create {primary_language} content with {translation_language} footnotes.

FORMAT:
Main text with markers[1]

---
[1] Translation
[2] Translation

CONTENT TO PROCESS:
{content}
""",
            "parallel": f"""
Create parallel {primary_language}/{translation_language} content.

FORMAT:
[{primary_language.upper()}]
Primary language content

[{translation_language.upper()}]
Translation

CONTENT TO PROCESS:
{content}
"""
        }
        
        prompt = formats.get(format, formats["inline"])
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
```

---

## Testing and Validation

### Language Output Test Suite

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class LanguageTestCase:
    name: str
    input_text: str
    expected_language: str
    input_language: str
    allow_mixed: bool = False

class LanguageOutputTester:
    """Test suite for multilingual output validation."""
    
    def __init__(self, generate_fn: Callable, detector: FastLanguageDetector):
        self.generate = generate_fn
        self.detector = detector
        self.results = []
    
    def run_test(self, test_case: LanguageTestCase) -> dict:
        """Run a single language output test."""
        
        # Generate output
        output = self.generate(test_case.input_text, test_case.expected_language)
        
        # Detect language
        detection = self.detector.detect(output)
        
        # Evaluate
        language_matches = (
            detection["language"] == test_case.expected_language or
            detection["language"] in test_case.expected_language.lower()
        )
        
        mixed_ok = test_case.allow_mixed or not detection["is_mixed"]
        
        passed = language_matches and mixed_ok
        
        result = {
            "test_name": test_case.name,
            "passed": passed,
            "expected_language": test_case.expected_language,
            "detected_language": detection["language"],
            "confidence": detection["confidence"],
            "is_mixed": detection["is_mixed"],
            "output_preview": output[:200] + "..." if len(output) > 200 else output
        }
        
        self.results.append(result)
        return result
    
    def run_suite(self, test_cases: list[LanguageTestCase]) -> dict:
        """Run all test cases."""
        
        for case in test_cases:
            self.run_test(case)
        
        passed = sum(1 for r in self.results if r["passed"])
        
        return {
            "total": len(test_cases),
            "passed": passed,
            "failed": len(test_cases) - passed,
            "pass_rate": passed / len(test_cases) if test_cases else 0,
            "results": self.results
        }
    
    def get_failure_report(self) -> str:
        """Generate report of failed tests."""
        
        failures = [r for r in self.results if not r["passed"]]
        
        if not failures:
            return "All tests passed!"
        
        report = "FAILED TESTS:\n\n"
        for f in failures:
            report += f"""
Test: {f["test_name"]}
Expected: {f["expected_language"]}
Detected: {f["detected_language"]} (confidence: {f["confidence"]:.2f})
Mixed: {f["is_mixed"]}
Output: {f["output_preview"]}
---
"""
        
        return report

# Example test cases
TEST_CASES = [
    LanguageTestCase(
        name="Japanese response to Japanese input",
        input_text="今日の天気はどうですか？",
        expected_language="japanese",
        input_language="japanese"
    ),
    LanguageTestCase(
        name="German response to English input",
        input_text="Explain quantum computing",
        expected_language="german",
        input_language="english"
    ),
    LanguageTestCase(
        name="Spanish with technical terms allowed",
        input_text="Describe API integration",
        expected_language="spanish",
        input_language="english",
        allow_mixed=True  # Allow English technical terms
    ),
    LanguageTestCase(
        name="Chinese response maintains language",
        input_text="请解释机器学习的基本概念",
        expected_language="chinese",
        input_language="chinese"
    ),
]
```

### Automated Quality Monitoring

```python
class LanguageQualityMonitor:
    """Monitor language quality in production."""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "language_violations": 0,
            "code_switching_incidents": 0,
            "corrections_applied": 0,
            "by_language": {}
        }
    
    def record_output(
        self,
        expected_language: str,
        detected_language: str,
        is_mixed: bool,
        was_corrected: bool
    ):
        """Record output for monitoring."""
        
        self.metrics["total_requests"] += 1
        
        if detected_language != expected_language:
            self.metrics["language_violations"] += 1
        
        if is_mixed:
            self.metrics["code_switching_incidents"] += 1
        
        if was_corrected:
            self.metrics["corrections_applied"] += 1
        
        # Track by language
        if expected_language not in self.metrics["by_language"]:
            self.metrics["by_language"][expected_language] = {
                "requests": 0,
                "violations": 0,
                "corrections": 0
            }
        
        lang_metrics = self.metrics["by_language"][expected_language]
        lang_metrics["requests"] += 1
        if detected_language != expected_language:
            lang_metrics["violations"] += 1
        if was_corrected:
            lang_metrics["corrections"] += 1
    
    def get_report(self) -> dict:
        """Generate quality report."""
        
        total = self.metrics["total_requests"]
        if total == 0:
            return {"error": "No data"}
        
        return {
            "summary": {
                "total_requests": total,
                "accuracy_rate": 1 - (self.metrics["language_violations"] / total),
                "code_switching_rate": self.metrics["code_switching_incidents"] / total,
                "correction_rate": self.metrics["corrections_applied"] / total
            },
            "by_language": {
                lang: {
                    "requests": data["requests"],
                    "accuracy": 1 - (data["violations"] / data["requests"]) if data["requests"] > 0 else 1,
                    "correction_rate": data["corrections"] / data["requests"] if data["requests"] > 0 else 0
                }
                for lang, data in self.metrics["by_language"].items()
            }
        }
```

---

## Hands-on Exercise

### Your Task

Build a robust multilingual output handler that:
1. Detects output language
2. Validates against expected language
3. Handles code-switching appropriately
4. Provides fallback correction

<details>
<summary>💡 Hints (click to expand)</summary>

- Use the `FastLanguageDetector` for efficiency
- Implement a configurable strategy pattern
- Include retry logic with exponential backoff

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from dataclasses import dataclass
from enum import Enum
import time

class CorrectionStrategy(Enum):
    STRICT = "strict"        # Always correct
    TOLERANT = "tolerant"    # Allow minor mixing
    PRESERVE = "preserve"    # Keep as-is, just validate

@dataclass
class OutputResult:
    text: str
    original_text: str
    detected_language: str
    expected_language: str
    was_corrected: bool
    correction_attempts: int
    code_switching_handled: bool
    quality_score: float

class RobustMultilingualHandler:
    """Production-ready multilingual output handler."""
    
    def __init__(
        self,
        client: OpenAI,
        strategy: CorrectionStrategy = CorrectionStrategy.TOLERANT,
        max_retries: int = 2,
        quality_threshold: float = 0.8
    ):
        self.client = client
        self.strategy = strategy
        self.max_retries = max_retries
        self.quality_threshold = quality_threshold
        self.detector = FastLanguageDetector()
        self.monitor = LanguageQualityMonitor()
    
    def handle_output(
        self,
        generated_text: str,
        expected_language: str,
        allow_technical_english: bool = True
    ) -> OutputResult:
        """Handle generated output with full validation and correction."""
        
        original_text = generated_text
        current_text = generated_text
        correction_attempts = 0
        
        # Step 1: Detect language
        detection = self.detector.detect(current_text)
        detected_language = detection["language"]
        is_mixed = detection["is_mixed"]
        
        # Step 2: Check if correction needed
        needs_correction = self._needs_correction(
            detected_language,
            expected_language,
            is_mixed,
            allow_technical_english
        )
        
        # Step 3: Apply correction if needed
        if needs_correction and self.strategy != CorrectionStrategy.PRESERVE:
            current_text, correction_attempts = self._apply_correction(
                current_text,
                expected_language,
                allow_technical_english
            )
            
            # Re-detect after correction
            detection = self.detector.detect(current_text)
            detected_language = detection["language"]
            is_mixed = detection["is_mixed"]
        
        # Step 4: Handle code-switching
        code_switching_handled = False
        if is_mixed and self.strategy == CorrectionStrategy.STRICT:
            current_text = self._handle_code_switching(
                current_text,
                expected_language
            )
            code_switching_handled = True
        
        # Step 5: Calculate quality score
        quality_score = self._calculate_quality(
            current_text,
            expected_language,
            detected_language
        )
        
        # Step 6: Record for monitoring
        self.monitor.record_output(
            expected_language=expected_language,
            detected_language=detected_language,
            is_mixed=is_mixed,
            was_corrected=correction_attempts > 0
        )
        
        return OutputResult(
            text=current_text,
            original_text=original_text,
            detected_language=detected_language,
            expected_language=expected_language,
            was_corrected=correction_attempts > 0,
            correction_attempts=correction_attempts,
            code_switching_handled=code_switching_handled,
            quality_score=quality_score
        )
    
    def _needs_correction(
        self,
        detected: str,
        expected: str,
        is_mixed: bool,
        allow_technical_english: bool
    ) -> bool:
        """Determine if correction is needed."""
        
        if detected == expected:
            if is_mixed and not allow_technical_english:
                return True
            return False
        
        return True
    
    def _apply_correction(
        self,
        text: str,
        target_language: str,
        allow_technical: bool
    ) -> tuple[str, int]:
        """Apply correction with retry logic."""
        
        attempts = 0
        current_text = text
        
        for attempt in range(self.max_retries):
            attempts += 1
            
            tech_instruction = """
Keep technical terms, code, and proper nouns in their original form.
""" if allow_technical else """
Translate ALL content including technical terms.
"""
            
            prompt = f"""
Rewrite this text entirely in {target_language}.
{tech_instruction}

TEXT:
{current_text}

OUTPUT: Text in {target_language} only.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            
            current_text = response.choices[0].message.content
            
            # Check if correction succeeded
            detection = self.detector.detect(current_text)
            if detection["language"] == target_language:
                break
            
            # Exponential backoff
            time.sleep(0.5 * (2 ** attempt))
        
        return current_text, attempts
    
    def _handle_code_switching(
        self,
        text: str,
        target_language: str
    ) -> str:
        """Eliminate code-switching."""
        
        prompt = f"""
Remove ALL code-switching from this text.
Convert everything to {target_language}.
Do not preserve any other language content.

TEXT: {text}

OUTPUT: Pure {target_language} text.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def _calculate_quality(
        self,
        text: str,
        expected: str,
        detected: str
    ) -> float:
        """Calculate language quality score."""
        
        detection = self.detector.detect(text)
        
        # Base score
        score = 1.0 if detected == expected else 0.5
        
        # Penalty for mixing
        if detection["is_mixed"]:
            score *= 0.8
        
        # Confidence factor
        score *= detection["confidence"]
        
        return min(1.0, max(0.0, score))

# Usage example
"""
handler = RobustMultilingualHandler(
    client=OpenAI(),
    strategy=CorrectionStrategy.TOLERANT,
    max_retries=2
)

result = handler.handle_output(
    generated_text="This is a test. これはテストです。",
    expected_language="japanese",
    allow_technical_english=True
)

print(f"Corrected: {result.was_corrected}")
print(f"Quality: {result.quality_score}")
print(f"Final text: {result.text}")
"""
```

</details>

---

## Summary

✅ **Always verify output language:** Don't assume models will comply
✅ **Use explicit language instructions:** System messages + output prefills
✅ **Detect code-switching:** It's common and needs explicit handling
✅ **Choose strategy based on context:** Strict for customer-facing, tolerant for internal
✅ **Monitor in production:** Track violations, corrections, and quality metrics
✅ **Fast detection for high volume:** Character-based heuristics work well

**Next:** [Lesson 14 - Next Topic](../14-next-topic/00-overview.md)

---

## Further Reading

- [Unicode Script Detection](https://www.unicode.org/reports/tr24/) - Technical spec
- [Language Identification Research](https://aclanthology.org/) - Academic papers
- [Anthropic Multilingual Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/multilingual-support) - Best practices

---

<!-- 
Sources Consulted:
- Unicode Technical Report #24: Script detection algorithms
- OpenAI API documentation: Output format constraints
- Anthropic multilingual support: Language enforcement patterns
-->
