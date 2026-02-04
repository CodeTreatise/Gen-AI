---
title: "Template Testing Frameworks"
---

# Template Testing Frameworks

## Introduction

Prompts are code. And like code, they need tests. Without testing, you'll discover prompt regressions in production, spend hours debugging subtle behavior changes, and hesitate to improve prompts for fear of breaking things. A template testing framework gives you confidence to iterate quickly while maintaining quality.

> **🔑 Key Insight:** Prompt testing isn't about verifying exact outputs—it's about validating structure, constraints, and behavior patterns that should remain consistent.

### What We'll Cover

- Unit tests for template rendering
- Variable validation testing
- Output structure validation
- Behavioral assertions
- Regression testing
- CI/CD integration
- LLM-based evaluation

### Prerequisites

- [Variable Substitution](./04-variable-substitution.md)
- [Template Versioning](./06-template-versioning.md)
- Python pytest basics

---

## Unit Testing Templates

### Testing Template Rendering

```python
import pytest
from string import Template

class PromptTemplate:
    """Simple prompt template for testing."""
    
    def __init__(self, template: str, required_vars: list[str]):
        self.template = Template(template)
        self.required_vars = set(required_vars)
    
    def render(self, **kwargs) -> str:
        missing = self.required_vars - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        return self.template.safe_substitute(**kwargs)

class TestPromptTemplateRendering:
    """Unit tests for template rendering."""
    
    @pytest.fixture
    def greeting_template(self):
        return PromptTemplate(
            "Hello $name, welcome to $company!",
            required_vars=["name", "company"]
        )
    
    def test_renders_with_all_variables(self, greeting_template):
        result = greeting_template.render(name="Alice", company="Acme")
        assert result == "Hello Alice, welcome to Acme!"
    
    def test_raises_on_missing_required_variable(self, greeting_template):
        with pytest.raises(ValueError) as exc_info:
            greeting_template.render(name="Alice")
        assert "company" in str(exc_info.value)
    
    def test_handles_special_characters_in_values(self, greeting_template):
        result = greeting_template.render(
            name="O'Brien",
            company="Ben & Jerry's"
        )
        assert "O'Brien" in result
        assert "Ben & Jerry's" in result
    
    def test_preserves_unknown_variables(self, greeting_template):
        """Safe substitute keeps unknown $variables."""
        template = PromptTemplate(
            "Hello $name, your code is $status",
            required_vars=["name"]
        )
        result = template.render(name="Alice")
        assert "$status" in result


class TestVariableValidation:
    """Test variable validation before rendering."""
    
    @pytest.fixture
    def validated_template(self):
        from dataclasses import dataclass
        
        @dataclass
        class VarSpec:
            name: str
            var_type: type
            required: bool = True
            min_length: int = None
            max_length: int = None
        
        class ValidatedTemplate:
            def __init__(self, template: str, specs: list[VarSpec]):
                self.template = Template(template)
                self.specs = {s.name: s for s in specs}
            
            def validate(self, **kwargs) -> list[str]:
                errors = []
                for name, spec in self.specs.items():
                    value = kwargs.get(name)
                    if value is None and spec.required:
                        errors.append(f"{name}: required")
                        continue
                    if value is not None:
                        if not isinstance(value, spec.var_type):
                            errors.append(f"{name}: expected {spec.var_type.__name__}")
                        if spec.min_length and len(str(value)) < spec.min_length:
                            errors.append(f"{name}: min length {spec.min_length}")
                        if spec.max_length and len(str(value)) > spec.max_length:
                            errors.append(f"{name}: max length {spec.max_length}")
                return errors
            
            def render(self, **kwargs) -> str:
                errors = self.validate(**kwargs)
                if errors:
                    raise ValueError(f"Validation failed: {errors}")
                return self.template.substitute(**kwargs)
        
        return ValidatedTemplate
    
    def test_validates_required_fields(self, validated_template):
        from dataclasses import dataclass
        
        @dataclass
        class VarSpec:
            name: str
            var_type: type
            required: bool = True
            min_length: int = None
            max_length: int = None
        
        template = validated_template(
            "Hello $name",
            [VarSpec("name", str, required=True)]
        )
        
        errors = template.validate()
        assert any("required" in e for e in errors)
    
    def test_validates_type(self, validated_template):
        from dataclasses import dataclass
        
        @dataclass
        class VarSpec:
            name: str
            var_type: type
            required: bool = True
            min_length: int = None
            max_length: int = None
        
        template = validated_template(
            "Count: $count",
            [VarSpec("count", int)]
        )
        
        errors = template.validate(count="not a number")
        assert any("expected int" in e for e in errors)
    
    def test_validates_length_constraints(self, validated_template):
        from dataclasses import dataclass
        
        @dataclass
        class VarSpec:
            name: str
            var_type: type
            required: bool = True
            min_length: int = None
            max_length: int = None
        
        template = validated_template(
            "Name: $name",
            [VarSpec("name", str, min_length=2, max_length=50)]
        )
        
        # Too short
        errors = template.validate(name="A")
        assert any("min length" in e for e in errors)
        
        # Too long
        errors = template.validate(name="A" * 100)
        assert any("max length" in e for e in errors)
```

---

## Output Structure Validation

### JSON Output Testing

```python
import json
import pytest
from jsonschema import validate, ValidationError

class TestOutputStructure:
    """Test that prompts produce valid structured output."""
    
    @pytest.fixture
    def classification_schema(self):
        return {
            "type": "object",
            "required": ["category", "confidence"],
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["billing", "technical", "general"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "reasoning": {
                    "type": "string"
                }
            },
            "additionalProperties": False
        }
    
    def test_valid_classification_output(self, classification_schema):
        output = {
            "category": "technical",
            "confidence": 0.95,
            "reasoning": "User mentioned login issues"
        }
        
        # Should not raise
        validate(instance=output, schema=classification_schema)
    
    def test_invalid_category_rejected(self, classification_schema):
        output = {
            "category": "invalid_category",
            "confidence": 0.9
        }
        
        with pytest.raises(ValidationError):
            validate(instance=output, schema=classification_schema)
    
    def test_confidence_bounds_enforced(self, classification_schema):
        # Over 1.0
        output = {"category": "billing", "confidence": 1.5}
        with pytest.raises(ValidationError):
            validate(instance=output, schema=classification_schema)
        
        # Under 0
        output = {"category": "billing", "confidence": -0.5}
        with pytest.raises(ValidationError):
            validate(instance=output, schema=classification_schema)


class OutputValidator:
    """Validate LLM outputs against expected structure."""
    
    def __init__(self, schema: dict):
        self.schema = schema
    
    def validate(self, output: str) -> dict:
        """Parse and validate output."""
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "error": f"Invalid JSON: {e}",
                "parsed": None
            }
        
        try:
            validate(instance=parsed, schema=self.schema)
            return {
                "valid": True,
                "error": None,
                "parsed": parsed
            }
        except ValidationError as e:
            return {
                "valid": False,
                "error": str(e.message),
                "parsed": parsed
            }

# Usage in tests
def test_llm_output_validation():
    validator = OutputValidator({
        "type": "object",
        "required": ["summary"],
        "properties": {
            "summary": {"type": "string", "minLength": 10}
        }
    })
    
    # Valid output
    result = validator.validate('{"summary": "This is a valid summary"}')
    assert result["valid"]
    
    # Invalid: too short
    result = validator.validate('{"summary": "Short"}')
    assert not result["valid"]
```

---

## Behavioral Assertions

### Pattern-Based Testing

```python
import re
from typing import Callable, List
from dataclasses import dataclass

@dataclass
class Assertion:
    name: str
    check: Callable[[str], bool]
    message: str

class BehavioralTester:
    """Test behavioral properties of prompt outputs."""
    
    def __init__(self):
        self.assertions: List[Assertion] = []
    
    def add_contains(self, text: str, case_sensitive: bool = False):
        """Assert output contains text."""
        def check(output: str) -> bool:
            if case_sensitive:
                return text in output
            return text.lower() in output.lower()
        
        self.assertions.append(Assertion(
            name=f"contains '{text}'",
            check=check,
            message=f"Output should contain '{text}'"
        ))
        return self
    
    def add_not_contains(self, text: str, case_sensitive: bool = False):
        """Assert output does not contain text."""
        def check(output: str) -> bool:
            if case_sensitive:
                return text not in output
            return text.lower() not in output.lower()
        
        self.assertions.append(Assertion(
            name=f"not contains '{text}'",
            check=check,
            message=f"Output should not contain '{text}'"
        ))
        return self
    
    def add_matches_pattern(self, pattern: str):
        """Assert output matches regex pattern."""
        compiled = re.compile(pattern)
        self.assertions.append(Assertion(
            name=f"matches pattern '{pattern}'",
            check=lambda o: bool(compiled.search(o)),
            message=f"Output should match pattern '{pattern}'"
        ))
        return self
    
    def add_length_between(self, min_len: int, max_len: int):
        """Assert output length is within range."""
        self.assertions.append(Assertion(
            name=f"length between {min_len}-{max_len}",
            check=lambda o: min_len <= len(o) <= max_len,
            message=f"Output length should be between {min_len} and {max_len}"
        ))
        return self
    
    def add_json_valid(self):
        """Assert output is valid JSON."""
        def check(output: str) -> bool:
            try:
                json.loads(output)
                return True
            except:
                return False
        
        self.assertions.append(Assertion(
            name="valid JSON",
            check=check,
            message="Output should be valid JSON"
        ))
        return self
    
    def add_custom(self, name: str, check: Callable[[str], bool], message: str):
        """Add custom assertion."""
        self.assertions.append(Assertion(name=name, check=check, message=message))
        return self
    
    def test(self, output: str) -> dict:
        """Run all assertions on output."""
        results = {
            "passed": [],
            "failed": []
        }
        
        for assertion in self.assertions:
            if assertion.check(output):
                results["passed"].append(assertion.name)
            else:
                results["failed"].append({
                    "name": assertion.name,
                    "message": assertion.message
                })
        
        results["all_passed"] = len(results["failed"]) == 0
        return results

# Usage
def test_support_response_behavior():
    tester = (
        BehavioralTester()
        .add_contains("thank you", case_sensitive=False)
        .add_not_contains("I don't know")
        .add_not_contains("as an AI")
        .add_length_between(50, 500)
        .add_matches_pattern(r"ticket #?\d+")
    )
    
    output = "Thank you for contacting us about ticket #12345. We'll look into this right away."
    
    results = tester.test(output)
    assert results["all_passed"], f"Failed: {results['failed']}"
```

### Constraint Testing

```python
class ConstraintTester:
    """Test that outputs respect specific constraints."""
    
    @staticmethod
    def test_no_pii(output: str) -> dict:
        """Check for potential PII in output."""
        patterns = {
            "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        }
        
        violations = []
        for pii_type, pattern in patterns.items():
            if re.search(pattern, output):
                violations.append(pii_type)
        
        return {
            "passed": len(violations) == 0,
            "violations": violations
        }
    
    @staticmethod
    def test_no_hallucination_markers(output: str) -> dict:
        """Check for common hallucination indicators."""
        markers = [
            "I don't have access to",
            "I cannot verify",
            "as of my last update",
            "I'm not sure, but",
            "I believe",  # uncertainty marker
        ]
        
        found = [m for m in markers if m.lower() in output.lower()]
        
        return {
            "passed": len(found) == 0,
            "markers_found": found
        }
    
    @staticmethod
    def test_professional_tone(output: str) -> dict:
        """Check for professional language."""
        unprofessional = [
            "lol", "omg", "wtf", "gonna", "wanna",
            "stuff", "things", "whatever", "idk"
        ]
        
        found = []
        for word in unprofessional:
            if re.search(rf"\b{word}\b", output, re.IGNORECASE):
                found.append(word)
        
        return {
            "passed": len(found) == 0,
            "unprofessional_terms": found
        }

# Usage in pytest
class TestOutputConstraints:
    
    def test_no_pii_leaked(self):
        output = "Your account has been updated. Thank you!"
        result = ConstraintTester.test_no_pii(output)
        assert result["passed"], f"PII found: {result['violations']}"
    
    def test_professional_tone(self):
        output = "I'll investigate this issue and get back to you."
        result = ConstraintTester.test_professional_tone(output)
        assert result["passed"], f"Unprofessional: {result['unprofessional_terms']}"
```

---

## Regression Testing

### Snapshot Testing

```python
import hashlib
from pathlib import Path
from typing import Optional

class PromptSnapshotTester:
    """Snapshot testing for prompt outputs."""
    
    def __init__(self, snapshot_dir: str = "snapshots"):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)
    
    def _get_snapshot_path(self, test_name: str) -> Path:
        return self.snapshot_dir / f"{test_name}.snapshot"
    
    def _hash_output(self, output: str) -> str:
        return hashlib.sha256(output.encode()).hexdigest()[:16]
    
    def assert_matches_snapshot(
        self,
        test_name: str,
        output: str,
        update: bool = False
    ) -> dict:
        """Assert output matches saved snapshot."""
        snapshot_path = self._get_snapshot_path(test_name)
        
        if update or not snapshot_path.exists():
            # Create/update snapshot
            snapshot_path.write_text(output)
            return {
                "status": "created" if not snapshot_path.exists() else "updated",
                "snapshot": output[:100] + "..."
            }
        
        # Compare with existing snapshot
        expected = snapshot_path.read_text()
        
        if output == expected:
            return {"status": "passed"}
        
        return {
            "status": "failed",
            "expected_hash": self._hash_output(expected),
            "actual_hash": self._hash_output(output),
            "diff_preview": self._generate_diff_preview(expected, output)
        }
    
    def _generate_diff_preview(self, expected: str, actual: str) -> str:
        """Generate a preview of differences."""
        import difflib
        diff = list(difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            lineterm=""
        ))
        return "\n".join(diff[:20])  # First 20 lines of diff

# Usage in tests
class TestPromptSnapshots:
    
    @pytest.fixture
    def snapshot_tester(self, tmp_path):
        return PromptSnapshotTester(str(tmp_path / "snapshots"))
    
    def test_greeting_prompt_snapshot(self, snapshot_tester):
        template = PromptTemplate(
            "Hello $name, welcome to $service!",
            ["name", "service"]
        )
        output = template.render(name="Test User", service="AI Assistant")
        
        result = snapshot_tester.assert_matches_snapshot(
            "greeting_standard",
            output
        )
        
        assert result["status"] in ["passed", "created"]
```

### Golden File Testing

```python
import yaml
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class GoldenTestCase:
    name: str
    inputs: Dict[str, any]
    expected_contains: List[str] = None
    expected_not_contains: List[str] = None
    expected_pattern: str = None

class GoldenFileTester:
    """Run tests from golden files (YAML test definitions)."""
    
    def __init__(self, template_renderer):
        self.renderer = template_renderer
    
    def load_test_cases(self, golden_file: str) -> List[GoldenTestCase]:
        """Load test cases from YAML file."""
        with open(golden_file) as f:
            data = yaml.safe_load(f)
        
        return [GoldenTestCase(**tc) for tc in data["test_cases"]]
    
    def run_tests(self, golden_file: str) -> dict:
        """Run all test cases from golden file."""
        test_cases = self.load_test_cases(golden_file)
        results = {"passed": 0, "failed": 0, "details": []}
        
        for tc in test_cases:
            output = self.renderer.render(**tc.inputs)
            passed = True
            errors = []
            
            if tc.expected_contains:
                for text in tc.expected_contains:
                    if text not in output:
                        passed = False
                        errors.append(f"Missing: '{text}'")
            
            if tc.expected_not_contains:
                for text in tc.expected_not_contains:
                    if text in output:
                        passed = False
                        errors.append(f"Should not contain: '{text}'")
            
            if tc.expected_pattern:
                if not re.search(tc.expected_pattern, output):
                    passed = False
                    errors.append(f"Pattern not found: '{tc.expected_pattern}'")
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            results["details"].append({
                "name": tc.name,
                "passed": passed,
                "errors": errors
            })
        
        return results

# golden_tests/greeting.yaml
"""
template_name: greeting
test_cases:
  - name: standard_greeting
    inputs:
      name: "Alice"
      company: "Acme"
    expected_contains:
      - "Hello Alice"
      - "Acme"
    expected_not_contains:
      - "undefined"
      
  - name: special_characters
    inputs:
      name: "O'Brien"
      company: "Ben & Jerry's"
    expected_contains:
      - "O'Brien"
      - "Ben & Jerry's"
"""
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test-prompts.yaml
name: Test Prompts

on:
  push:
    paths:
      - 'prompts/**'
      - 'tests/**'
  pull_request:
    paths:
      - 'prompts/**'
      - 'tests/**'

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install pytest pytest-cov pyyaml jsonschema
      
      - name: Validate prompt schemas
        run: python scripts/validate_schemas.py
      
      - name: Run unit tests
        run: pytest tests/unit -v --cov=prompts
      
      - name: Run regression tests
        run: pytest tests/regression -v
      
      - name: Run golden file tests
        run: pytest tests/golden -v
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Pre-Commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-prompts
        name: Validate Prompt Files
        entry: python scripts/validate_prompts.py
        language: python
        files: prompts/.*\.yaml$
        
      - id: test-prompts
        name: Test Prompts
        entry: pytest tests/unit -x
        language: python
        pass_filenames: false
```

### Validation Script

```python
#!/usr/bin/env python
# scripts/validate_prompts.py
import sys
import yaml
import json
from pathlib import Path
from jsonschema import validate, ValidationError

PROMPT_SCHEMA = {
    "type": "object",
    "required": ["name", "version", "template"],
    "properties": {
        "name": {"type": "string", "pattern": "^[a-z0-9-]+$"},
        "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
        "description": {"type": "string"},
        "template": {"type": "string", "minLength": 10},
        "variables": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "type"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "required": {"type": "boolean"}
                }
            }
        }
    }
}

def validate_prompt_file(path: Path) -> list:
    """Validate a single prompt file."""
    errors = []
    
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]
    
    try:
        validate(instance=data, schema=PROMPT_SCHEMA)
    except ValidationError as e:
        errors.append(f"Schema validation: {e.message}")
    
    # Additional checks
    if "template" in data:
        # Check for unmatched variables
        import re
        template_vars = set(re.findall(r'\$(\w+)', data["template"]))
        defined_vars = set(v["name"] for v in data.get("variables", []))
        
        undefined = template_vars - defined_vars
        if undefined:
            errors.append(f"Undefined variables in template: {undefined}")
    
    return errors

def main():
    prompts_dir = Path("prompts")
    all_errors = {}
    
    for path in prompts_dir.rglob("*.yaml"):
        errors = validate_prompt_file(path)
        if errors:
            all_errors[str(path)] = errors
    
    if all_errors:
        print("❌ Validation failed:")
        for path, errors in all_errors.items():
            print(f"\n{path}:")
            for error in errors:
                print(f"  - {error}")
        sys.exit(1)
    
    print("✅ All prompts valid")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

---

## LLM-Based Evaluation

### Using LLM as Judge

```python
from openai import OpenAI

class LLMEvaluator:
    """Use LLM to evaluate prompt outputs."""
    
    EVALUATION_PROMPT = """
You are evaluating an AI assistant's response.

Criteria:
1. Helpfulness (1-5): Does it address the user's needs?
2. Accuracy (1-5): Is the information correct?
3. Tone (1-5): Is it professional and appropriate?
4. Completeness (1-5): Does it fully answer the question?

User Query: {query}

Assistant Response: {response}

Evaluate the response. Return JSON:
{{
    "helpfulness": <1-5>,
    "accuracy": <1-5>,
    "tone": <1-5>,
    "completeness": <1-5>,
    "overall": <1-5>,
    "feedback": "<brief feedback>"
}}
"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def evaluate(self, query: str, response: str) -> dict:
        """Evaluate a response using LLM."""
        result = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": self.EVALUATION_PROMPT.format(
                    query=query,
                    response=response
                )
            }],
            response_format={"type": "json_object"}
        )
        
        return json.loads(result.choices[0].message.content)
    
    def batch_evaluate(
        self, 
        test_cases: list[dict],
        threshold: float = 3.5
    ) -> dict:
        """Evaluate multiple test cases."""
        results = {
            "passed": 0,
            "failed": 0,
            "evaluations": []
        }
        
        for tc in test_cases:
            eval_result = self.evaluate(tc["query"], tc["response"])
            passed = eval_result["overall"] >= threshold
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            results["evaluations"].append({
                "query": tc["query"][:50] + "...",
                "scores": eval_result,
                "passed": passed
            })
        
        return results
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Test template rendering | Catch variable issues early |
| Validate output structure | Ensure downstream compatibility |
| Use behavioral assertions | Verify intent, not exact text |
| Run tests in CI | Catch regressions automatically |
| Keep golden files updated | Maintain known-good baselines |
| Combine rule + LLM testing | Rules for structure, LLM for quality |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Testing exact output | Brittle tests | Use behavioral assertions |
| No variable validation | Runtime errors | Validate before render |
| Missing CI integration | Manual testing forgotten | Automate in pipeline |
| Outdated snapshots | False positives | Review and update regularly |
| Over-relying on LLM eval | Expensive, slow | Use for sampling, not all tests |

---

## Hands-on Exercise

### Your Task

Build a test suite for a customer support prompt template that:
1. Tests variable validation (required fields, types)
2. Validates output structure (JSON with specific fields)
3. Checks behavioral constraints (professional tone, no PII)
4. Includes at least 3 golden test cases

### Requirements

1. Create the prompt template with variables: `customer_name`, `issue`, `priority`
2. Write 5+ unit tests for rendering
3. Write 3+ behavioral tests
4. Create a YAML golden file with test cases

<details>
<summary>💡 Hints</summary>

- Use pytest fixtures for template setup
- Combine multiple assertion types
- Test edge cases (empty strings, special characters)
- Include both passing and failing golden cases

</details>

<details>
<summary>✅ Solution</summary>

**test_support_template.py:**
```python
import pytest
from string import Template

class SupportTemplate:
    def __init__(self):
        self.template = Template("""
Customer: $customer_name
Issue: $issue
Priority: $priority

Please respond professionally and helpfully.
""")
        self.required = ["customer_name", "issue", "priority"]
    
    def render(self, **kwargs):
        missing = set(self.required) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing: {missing}")
        return self.template.substitute(**kwargs)

class TestSupportTemplate:
    @pytest.fixture
    def template(self):
        return SupportTemplate()
    
    def test_renders_all_fields(self, template):
        result = template.render(
            customer_name="Alice",
            issue="Login problem",
            priority="high"
        )
        assert "Alice" in result
        assert "Login problem" in result
        assert "high" in result
    
    def test_missing_required_raises(self, template):
        with pytest.raises(ValueError):
            template.render(customer_name="Alice")
    
    def test_special_characters(self, template):
        result = template.render(
            customer_name="O'Brien",
            issue="Can't login",
            priority="high"
        )
        assert "O'Brien" in result

class TestBehavioralConstraints:
    def test_professional_tone(self):
        output = "Thank you for contacting support."
        unprofessional = ["lol", "omg", "gonna"]
        for word in unprofessional:
            assert word not in output.lower()
    
    def test_no_pii(self):
        output = "We'll help with your account."
        assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", output)  # No SSN
```

</details>

---

## Summary

- Unit test template rendering and variable validation
- Validate output structure with JSON schemas
- Use behavioral assertions for flexible testing
- Implement regression testing with snapshots
- Integrate tests into CI/CD pipeline
- Combine rule-based and LLM-based evaluation

**Previous:** [Template Versioning](./06-template-versioning.md)

**Back to:** [Prompt Templates Overview](./00-prompt-templates-overview.md)

---

<!-- Sources: pytest documentation, software testing best practices, LLM evaluation patterns -->
