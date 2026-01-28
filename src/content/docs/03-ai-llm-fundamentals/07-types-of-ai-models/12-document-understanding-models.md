---
title: "Document Understanding Models"
---

# Document Understanding Models

## Introduction

Document understanding models process PDFs, images of documents, and complex layouts. They extract text, tables, charts, and understand document structure.

### What We'll Cover

- Native PDF processing
- Layout understanding
- Table and chart extraction
- Multi-page analysis

---

## Native PDF Processing

### Claude PDF Support

```python
from anthropic import Anthropic
import base64

client = Anthropic()

def process_pdf(pdf_path: str, prompt: str) -> str:
    """Process PDF natively with Claude"""
    
    with open(pdf_path, "rb") as f:
        pdf_data = base64.b64encode(f.read()).decode()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }]
    )
    
    return response.content[0].text

# Example: Analyze a research paper
result = process_pdf(
    "research_paper.pdf",
    """Analyze this paper and provide:
    1. Title and authors
    2. Abstract summary
    3. Key findings
    4. Methodology overview
    5. Conclusions"""
)
```

### Gemini PDF Processing

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")

def analyze_pdf_gemini(pdf_path: str, prompt: str) -> str:
    """Process PDF with Gemini"""
    
    # Upload file
    pdf_file = genai.upload_file(pdf_path)
    
    response = model.generate_content([pdf_file, prompt])
    
    return response.text

# Analyze financial report
result = analyze_pdf_gemini(
    "quarterly_report.pdf",
    "Extract all financial metrics and summarize performance"
)
```

---

## Layout Understanding

### Document Structure Analysis

```python
def analyze_document_structure(pdf_path: str) -> dict:
    """Understand document layout and structure"""
    
    with open(pdf_path, "rb") as f:
        pdf_data = base64.b64encode(f.read()).decode()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data
                    }
                },
                {
                    "type": "text",
                    "text": """Analyze the document structure:
                    
Return JSON with:
- document_type: (report, form, contract, article, etc.)
- sections: list of section headings
- has_tables: boolean
- has_charts: boolean
- has_images: boolean
- page_count: number
- layout_type: (single-column, multi-column, mixed)"""
                }
            ]
        }],
        response_format={"type": "json_object"} if hasattr(client, 'beta') else None
    )
    
    import json
    try:
        return json.loads(response.content[0].text)
    except:
        return {"raw": response.content[0].text}
```

### Form Field Extraction

```python
def extract_form_fields(form_image: str) -> dict:
    """Extract form fields and values"""
    
    with open(form_image, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": """Extract all form fields and their values.
                    
Return JSON:
{
    "form_type": "...",
    "fields": [
        {"label": "...", "value": "...", "type": "text/checkbox/date/etc"}
    ]
}"""
                }
            ]
        }]
    )
    
    import json
    return json.loads(response.content[0].text)

# Extract from scanned form
fields = extract_form_fields("application_form.png")
```

---

## Table and Chart Extraction

### Table Extraction

```python
def extract_tables(document_path: str) -> list:
    """Extract all tables from document"""
    
    with open(document_path, "rb") as f:
        doc_data = base64.b64encode(f.read()).decode()
    
    media_type = "application/pdf" if document_path.endswith(".pdf") else "image/png"
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document" if media_type == "application/pdf" else "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": doc_data
                    }
                },
                {
                    "type": "text",
                    "text": """Extract ALL tables from this document.
                    
For each table, return:
- table_number
- title (if any)
- headers (column names)
- rows (as array of arrays)
- page_number

Return as JSON array."""
                }
            ]
        }]
    )
    
    import json
    return json.loads(response.content[0].text)

# Extract financial tables
tables = extract_tables("annual_report.pdf")
for table in tables:
    print(f"Table {table['table_number']}: {table.get('title', 'Untitled')}")
```

### Chart Analysis

```python
def analyze_chart(chart_image: str) -> dict:
    """Extract data and insights from chart"""
    
    with open(chart_image, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": """Analyze this chart:

Return JSON with:
- chart_type: (bar, line, pie, scatter, etc.)
- title: chart title
- x_axis_label: 
- y_axis_label:
- data_points: extracted values where readable
- trends: observed trends
- insights: key takeaways"""
                }
            ]
        }]
    )
    
    import json
    return json.loads(response.content[0].text)
```

---

## Multi-Page Analysis

### Process Long Documents

```python
def analyze_multi_page_document(
    pdf_path: str,
    analysis_type: str = "summary"
) -> dict:
    """Analyze multi-page documents comprehensively"""
    
    with open(pdf_path, "rb") as f:
        pdf_data = base64.b64encode(f.read()).decode()
    
    prompts = {
        "summary": "Provide a comprehensive summary of this document.",
        "extract": "Extract all key information, data points, and facts.",
        "contract": """Analyze this contract:
            - Parties involved
            - Key terms and conditions
            - Important dates
            - Obligations of each party
            - Termination clauses
            - Potential risks""",
        "research": """Analyze this research paper:
            - Research question
            - Methodology
            - Key findings
            - Limitations
            - Conclusions
            - Citation-worthy points"""
    }
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data
                    }
                },
                {"type": "text", "text": prompts.get(analysis_type, prompts["summary"])}
            ]
        }]
    )
    
    return {
        "analysis_type": analysis_type,
        "result": response.content[0].text
    }
```

### Page-by-Page Processing

```python
import fitz  # PyMuPDF

def process_pages_individually(pdf_path: str) -> list:
    """Process each page and aggregate results"""
    
    doc = fitz.open(pdf_path)
    results = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Render page as image
        pix = page.get_pixmap(dpi=150)
        img_data = base64.b64encode(pix.tobytes("png")).decode()
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_data
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Extract all text and describe any images/tables on this page (page {page_num + 1})"
                    }
                ]
            }]
        )
        
        results.append({
            "page": page_num + 1,
            "content": response.content[0].text
        })
    
    doc.close()
    return results
```

---

## Specialized Document Types

### Invoice Processing

```python
def process_invoice(invoice_path: str) -> dict:
    """Extract invoice information"""
    
    response = process_document_with_schema(
        invoice_path,
        """Extract invoice details:
        - invoice_number
        - date
        - due_date
        - vendor_name
        - vendor_address
        - customer_name
        - customer_address
        - line_items: [{description, quantity, unit_price, total}]
        - subtotal
        - tax
        - total
        - payment_terms"""
    )
    
    return response
```

### Resume Parsing

```python
def parse_resume(resume_path: str) -> dict:
    """Parse resume into structured data"""
    
    with open(resume_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": data
                    }
                },
                {
                    "type": "text",
                    "text": """Parse this resume into JSON:
{
    "name": "",
    "contact": {"email": "", "phone": "", "location": ""},
    "summary": "",
    "experience": [{"company": "", "title": "", "dates": "", "highlights": []}],
    "education": [{"institution": "", "degree": "", "year": ""}],
    "skills": [],
    "certifications": []
}"""
                }
            ]
        }]
    )
    
    import json
    return json.loads(response.content[0].text)
```

---

## Hands-on Exercise

### Your Task

Build a document processor:

```python
from anthropic import Anthropic
import base64
from pathlib import Path

client = Anthropic()

class DocumentProcessor:
    """Comprehensive document processing"""
    
    def __init__(self):
        self.supported_types = [".pdf", ".png", ".jpg", ".jpeg"]
    
    def process(self, file_path: str, task: str = "summarize") -> dict:
        """Process document with specified task"""
        
        path = Path(file_path)
        if path.suffix.lower() not in self.supported_types:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        
        media_type = self._get_media_type(path.suffix)
        content_type = "document" if path.suffix == ".pdf" else "image"
        
        tasks = {
            "summarize": "Summarize this document concisely.",
            "extract_text": "Extract all text from this document.",
            "extract_tables": "Extract all tables as JSON arrays.",
            "key_points": "List the key points from this document.",
            "qa_prep": "Prepare Q&A pairs based on this document content."
        }
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": content_type,
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data
                        }
                    },
                    {"type": "text", "text": tasks.get(task, task)}
                ]
            }]
        )
        
        return {
            "file": file_path,
            "task": task,
            "result": response.content[0].text
        }
    
    def _get_media_type(self, suffix: str) -> str:
        types = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg"
        }
        return types.get(suffix.lower(), "application/octet-stream")

# Usage
processor = DocumentProcessor()

# Summarize PDF
# result = processor.process("report.pdf", "summarize")
# print(result["result"])

# Extract tables
# result = processor.process("data.pdf", "extract_tables")
```

---

## Summary

✅ **Claude & Gemini**: Native PDF support

✅ **Layout understanding**: Structure, sections, formatting

✅ **Table extraction**: Convert tables to structured data

✅ **Chart analysis**: Extract data and insights

✅ **Multi-page**: Handle long documents effectively

**Next:** [Agent & Tool-Use Models](./13-agent-tool-use-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Multimodal Models](./11-multimodal-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Agent & Tool-Use](./13-agent-tool-use-models.md) |

