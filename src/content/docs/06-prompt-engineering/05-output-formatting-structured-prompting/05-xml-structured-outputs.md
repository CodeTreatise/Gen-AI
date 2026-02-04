---
title: "XML & Structured Text Outputs"
---

# XML & Structured Text Outputs

## Introduction

XML outputs provide a middle ground between JSON's strict structure and markdown's readability. XML excels at representing hierarchical data with mixed content—text that contains both prose and structured elements. This lesson covers when to use XML, how to define tag structures, and parsing considerations.

> **🤖 AI Context:** XML is particularly useful when you need to extract multiple distinct pieces of information from a single response, or when outputs contain both structured data and natural language explanations.

### What We'll Cover

- When to use XML over JSON or markdown
- Defining XML tag structures
- Attributes vs nested elements
- Mixed content handling
- Parsing XML outputs

### Prerequisites

- [Markdown Formatting Requests](./04-markdown-formatting-requests.md)

---

## When to Use XML

### XML vs JSON vs Markdown

| Scenario | Best Format | Why |
|----------|-------------|-----|
| API data exchange | JSON | Universal parsing, typed |
| Human-readable docs | Markdown | Renders beautifully |
| Mixed content | XML | Text + structured data |
| Multi-part extraction | XML | Clear section boundaries |
| Pipeline processing | XML | Easy section extraction |
| Reasoning + result | XML | Separate thought from answer |

### XML Strengths

```
✅ Clear boundaries with opening/closing tags
✅ Supports mixed content (text + data)
✅ Self-documenting with meaningful tag names
✅ Attributes for metadata
✅ Easy to extract specific sections
```

---

## Defining XML Tag Structures

### Basic Tag Definition

```markdown
# Output Format

Wrap your response in XML tags:

<analysis>
  <summary>2-3 sentence summary</summary>
  <sentiment>positive, negative, or neutral</sentiment>
  <confidence>0.0 to 1.0</confidence>
  <key_points>
    <point>First key point</point>
    <point>Second key point</point>
  </key_points>
</analysis>
```

### Tag Naming Conventions

| Pattern | Example | Use Case |
|---------|---------|----------|
| snake_case | `<key_points>` | General use |
| camelCase | `<keyPoints>` | JavaScript integration |
| kebab-case | `<key-points>` | HTML-like |
| Single word | `<summary>` | Simple elements |

```markdown
# Tag Naming

Use snake_case for all tag names:
- <user_input> not <userInput> or <UserInput>
- <analysis_result> not <analysisResult>
- <key_points> not <keyPoints>
```

---

## Attributes vs Nested Elements

### When to Use Attributes

Attributes work best for metadata:

```xml
<document id="doc-123" version="2" language="en">
  <title>Understanding AI</title>
  <content>...</content>
</document>
```

### When to Use Nested Elements

Nested elements work best for content:

```xml
<review>
  <text>Great product, highly recommended!</text>
  <rating>5</rating>
  <author>John Doe</author>
</review>
```

### Comparison

| Use Attributes For | Use Elements For |
|-------------------|------------------|
| IDs and references | Content/text |
| Metadata | Complex nested data |
| Single values | Multiple values |
| Non-repeated info | Lists/arrays |

### Combined Approach

```xml
<example id="1" type="positive">
  <input>I love this product!</input>
  <output category="sentiment">positive</output>
  <confidence level="high">0.95</confidence>
</example>
```

---

## Mixed Content

XML's key advantage is handling mixed content—text with embedded structured elements:

### Prose with Annotations

```markdown
# Output Format

Provide analysis with inline annotations:

<analysis>
  The company reported <metric type="revenue">$5.2 billion</metric> in 
  quarterly revenue, representing a <metric type="growth">12%</metric> 
  increase year-over-year. However, <concern priority="high">profit 
  margins declined</concern> due to increased operational costs.
</analysis>
```

### Document with Embedded Structure

```xml
<response>
  <reasoning>
    I analyzed the customer's request and identified the main issue.
    The error code <code>ERR_401</code> indicates an authentication problem.
  </reasoning>
  
  <answer>
    Your session has expired. Please <action>log in again</action> to 
    continue using the service.
  </answer>
  
  <metadata>
    <category>authentication</category>
    <confidence>0.92</confidence>
  </metadata>
</response>
```

---

## Common XML Patterns

### Reasoning + Result Pattern

Separate the model's thinking from the final answer:

```markdown
# Output Format

<response>
  <thinking>
    [Your step-by-step reasoning here]
  </thinking>
  <answer>
    [Your final answer here]
  </answer>
</response>
```

### Multi-Section Analysis

```xml
<analysis>
  <summary>
    Brief overview of findings
  </summary>
  
  <sections>
    <section name="strengths">
      <point>First strength</point>
      <point>Second strength</point>
    </section>
    <section name="weaknesses">
      <point>First weakness</point>
    </section>
    <section name="recommendations">
      <point priority="high">Top recommendation</point>
      <point priority="medium">Secondary recommendation</point>
    </section>
  </sections>
  
  <conclusion>
    Final thoughts and next steps
  </conclusion>
</analysis>
```

### Extraction with Confidence

```xml
<extraction>
  <entity type="person" confidence="0.95">John Smith</entity>
  <entity type="organization" confidence="0.88">TechCorp Inc.</entity>
  <entity type="date" confidence="0.99">January 15, 2025</entity>
  <entity type="amount" confidence="0.72">approximately $50,000</entity>
</extraction>
```

---

## Parsing XML Outputs

### Python Parsing

```python
import xml.etree.ElementTree as ET

def parse_analysis(response_text):
    # Extract XML from response
    root = ET.fromstring(response_text)
    
    return {
        'summary': root.find('summary').text,
        'sentiment': root.find('sentiment').text,
        'confidence': float(root.find('confidence').text),
        'points': [p.text for p in root.findall('.//point')]
    }
```

### JavaScript Parsing

```javascript
function parseAnalysis(responseText) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(responseText, 'text/xml');
  
  return {
    summary: doc.querySelector('summary').textContent,
    sentiment: doc.querySelector('sentiment').textContent,
    confidence: parseFloat(doc.querySelector('confidence').textContent),
    points: Array.from(doc.querySelectorAll('point')).map(p => p.textContent)
  };
}
```

### Regex Extraction (Simple Cases)

```python
import re

def extract_answer(response):
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
```

---

## XML Output Specification

### Complete Specification Example

```markdown
# Output Format

Return your response as XML with this structure:

## Required Elements

<response>
  <summary>String: 2-3 sentence summary</summary>
  <analysis>String: Detailed analysis</analysis>
  <recommendations>
    <item priority="high|medium|low">Recommendation text</item>
    <!-- 3-5 items required -->
  </recommendations>
  <metadata>
    <confidence>Number: 0.0 to 1.0</confidence>
    <category>String: One of "technical", "business", "general"</category>
  </metadata>
</response>

## Rules

1. All elements are required unless marked optional
2. Use only the specified priority values
3. Include 3-5 recommendation items
4. Confidence must be a decimal number
5. No text outside the <response> tags
```

---

## Handling Special Characters

### XML Entities

```markdown
# Character Encoding

Use XML entities for special characters:
- &lt; for <
- &gt; for >
- &amp; for &
- &quot; for "
- &apos; for '

Example: "Revenue increased by 5% & profits rose" becomes:
<text>Revenue increased by 5% &amp; profits rose</text>
```

### CDATA for Raw Content

```markdown
# Code Content

For code or raw text, use CDATA sections:

<code_snippet>
<![CDATA[
function example() {
  if (x < 10 && y > 5) {
    return x + y;
  }
}
]]>
</code_snippet>
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Consistent tag naming | Easier parsing |
| Meaningful tag names | Self-documenting |
| Close all tags | Valid XML required |
| Use attributes for metadata | Cleaner structure |
| Specify expected content | Predictable outputs |
| Handle special characters | Prevent parsing errors |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Unclosed tags | Require complete XML |
| Inconsistent tag names | Specify exact names |
| Special chars in text | Request entity encoding |
| Deep nesting | Limit to 3-4 levels |
| Mixing with markdown | Keep XML separate |

---

## Hands-on Exercise

### Your Task

Create an XML output specification for a document analysis system.

### Requirements

1. Extract: title, author, date, summary, key topics, sentiment
2. Key topics should have relevance scores
3. Include a confidence score for each extracted field
4. Handle missing information with empty tags or attributes

### Sample Document

```
"The Impact of AI on Healthcare" by Dr. Sarah Chen
Published: March 2024

This paper examines how artificial intelligence is transforming 
diagnostic medicine, particularly in radiology and pathology...
```

<details>
<summary>💡 Hints (click to expand)</summary>

- How do you represent confidence per field?
- Should missing author be an empty tag or omitted?
- How do you structure topics with scores?

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```markdown
# Instructions

Extract document metadata and analysis, returning structured XML.

# Output Format

```xml
<document_analysis>
  <metadata>
    <title confidence="0.0-1.0">Extracted title or empty if none</title>
    <author confidence="0.0-1.0">Author name or empty if none</author>
    <date confidence="0.0-1.0" format="YYYY-MM">Publication date or empty</date>
  </metadata>
  
  <summary confidence="0.0-1.0">
    2-3 sentence summary of the document
  </summary>
  
  <topics>
    <topic relevance="0.0-1.0">Topic 1</topic>
    <topic relevance="0.0-1.0">Topic 2</topic>
    <!-- 3-5 topics -->
  </topics>
  
  <sentiment>
    <classification>positive | negative | neutral | mixed</classification>
    <confidence>0.0-1.0</confidence>
    <explanation>Brief explanation of sentiment assessment</explanation>
  </sentiment>
</document_analysis>
```

# Rules

1. Include confidence attribute on all extracted metadata fields
2. Use empty tags (e.g., <author></author>) for missing information
3. Extract 3-5 key topics with relevance scores
4. Relevance and confidence scores between 0.0 and 1.0
5. Return only the XML, no other text
```

**Expected output:**
```xml
<document_analysis>
  <metadata>
    <title confidence="0.98">The Impact of AI on Healthcare</title>
    <author confidence="0.95">Dr. Sarah Chen</author>
    <date confidence="0.90" format="YYYY-MM">2024-03</date>
  </metadata>
  
  <summary confidence="0.88">
    This academic paper explores the transformative effects of artificial 
    intelligence on the healthcare industry. It specifically focuses on 
    AI applications in diagnostic medicine, with emphasis on radiology 
    and pathology departments.
  </summary>
  
  <topics>
    <topic relevance="0.95">artificial intelligence in healthcare</topic>
    <topic relevance="0.92">diagnostic medicine</topic>
    <topic relevance="0.88">radiology AI applications</topic>
    <topic relevance="0.85">pathology automation</topic>
  </topics>
  
  <sentiment>
    <classification>positive</classification>
    <confidence>0.75</confidence>
    <explanation>The document appears to view AI healthcare applications 
    favorably, using terms like "transforming" which suggests positive 
    change.</explanation>
  </sentiment>
</document_analysis>
```

</details>

### Bonus Challenge

- [ ] Add a `citations` section for referenced works
- [ ] Include `reading_level` assessment (academic, professional, general)

---

## Summary

✅ **XML excels** at mixed content and multi-section outputs

✅ **Use attributes** for metadata, elements for content

✅ **Consistent naming** enables reliable parsing

✅ **Handle special characters** with entities or CDATA

✅ **Clear specifications** produce valid, parseable XML

**Next:** [Schema Definition in Prompts](./06-schema-definition.md)

---

## Further Reading

- [XML Specification (W3C)](https://www.w3.org/TR/xml/)
- [Python xml.etree.ElementTree](https://docs.python.org/3/library/xml.etree.elementtree.html)
- [JavaScript DOMParser](https://developer.mozilla.org/en-US/docs/Web/API/DOMParser)

---

<!-- 
Sources Consulted:
- W3C XML Specification: https://www.w3.org/TR/xml/
- OpenAI Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
-->
