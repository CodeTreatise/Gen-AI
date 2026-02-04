---
title: "Document Understanding"
---

# Document Understanding

## Introduction

Documents—PDFs, invoices, forms, academic papers—contain structured information that's valuable for automation. This lesson covers how to extract data from documents using multimodal AI, from simple text extraction to complex table parsing and form field detection.

> **🔑 Key Insight:** Gemini supports native PDF processing up to 1,000 pages, while OpenAI and Anthropic require rendering pages as images. Choose the right approach based on your document type and volume.

### What We'll Cover

- PDF processing strategies by provider
- Table and chart extraction techniques
- Form field detection and data extraction
- OCR from imperfect images
- Structured output from unstructured documents

### Prerequisites

- [Image Prompting Fundamentals](./01-image-prompting.md)

---

## PDF Processing Strategies

### Gemini: Native PDF Support

Gemini can process PDFs directly without converting to images:

```python
from google import genai
from google.genai import types

client = genai.Client()

# Method 1: Local PDF file (small files)
with open("contract.pdf", "rb") as f:
    pdf_bytes = f.read()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
        "Extract all parties, dates, and key obligations from this contract."
    ]
)

print(response.text)
```

**Gemini Files API for Large PDFs:**

```python
from google import genai

client = genai.Client()

# Upload PDF (stored 48 hours)
uploaded_pdf = client.files.upload(file="annual_report.pdf")

# Wait for processing if needed
import time
while uploaded_pdf.state.name == "PROCESSING":
    time.sleep(2)
    uploaded_pdf = client.files.get(name=uploaded_pdf.name)

# Use in request
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        uploaded_pdf,
        "Summarize the financial highlights from this annual report."
    ]
)
```

### PDF Limits

| Provider | Format | Max Pages | Max Size | Tokens/Page |
|----------|--------|-----------|----------|-------------|
| Gemini | Native PDF | 1,000 | 50MB | ~258 |
| OpenAI | Image per page | Unlimited* | 50MB total | 85-765 |
| Anthropic | Image per page | 100 pages | 32MB total | varies |

*Limited by total payload size

### OpenAI/Anthropic: PDF to Images

When native PDF isn't available, convert to images:

```python
import fitz  # PyMuPDF
import base64
from io import BytesIO
from openai import OpenAI

def pdf_to_images(pdf_path: str, max_pages: int = 20, dpi: int = 150) -> list:
    """Convert PDF pages to base64 images."""
    
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(min(len(doc), max_pages)):
        page = doc[page_num]
        # Render at specified DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        b64_image = base64.standard_b64encode(img_bytes).decode()
        
        images.append({
            "page_number": page_num + 1,
            "base64": b64_image,
            "width": pix.width,
            "height": pix.height
        })
    
    doc.close()
    return images

client = OpenAI()

# Process PDF
pages = pdf_to_images("invoice.pdf", max_pages=5)

# Build message with all pages
content = [{"type": "input_text", "text": "Extract invoice details from these pages:"}]

for page in pages:
    content.append({
        "type": "input_image",
        "image_url": f"data:image/png;base64,{page['base64']}",
        "detail": "high"
    })
    content.append({
        "type": "input_text",
        "text": f"[Page {page['page_number']}]"
    })

content.append({
    "type": "input_text",
    "text": "Return: invoice number, date, vendor, line items, and total."
})

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{"role": "user", "content": content}]
)
```

---

## Table Extraction

### Simple Tables

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class TableRow(BaseModel):
    product: str
    quantity: int
    unit_price: float
    total: float

class ExtractedTable(BaseModel):
    headers: list[str]
    rows: list[TableRow]
    table_total: float | None

response = client.responses.parse(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {
                "type": "input_image",
                "image_url": table_image_url,
                "detail": "high"
            },
            {
                "type": "input_text",
                "text": "Extract the product table from this image."
            }
        ]
    }],
    text_format=ExtractedTable
)

table = response.output_parsed
for row in table.rows:
    print(f"{row.product}: {row.quantity} × ${row.unit_price} = ${row.total}")
```

### Complex Multi-Section Tables

```python
from pydantic import BaseModel

class FinancialStatement(BaseModel):
    section: str
    line_items: list[dict]  # {name: str, values: list[float]}
    section_total: float | None

class AnnualReportTables(BaseModel):
    income_statement: FinancialStatement | None
    balance_sheet: FinancialStatement | None
    cash_flow: FinancialStatement | None
    fiscal_year: str
    currency: str

prompt = """
Extract financial tables from this annual report page.
Look for:
1. Income Statement / P&L
2. Balance Sheet
3. Cash Flow Statement

For each table found, extract:
- Section name
- All line items with their values
- Section totals
"""

response = client.responses.parse(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": report_page, "detail": "high"},
            {"type": "input_text", "text": prompt}
        ]
    }],
    text_format=AnnualReportTables
)
```

### Table Extraction Best Practices

| Challenge | Solution |
|-----------|----------|
| Merged cells | Ask model to "note any merged or spanning cells" |
| Multi-page tables | Process pages together, ask for "continuation" |
| Nested headers | Request hierarchical header extraction |
| Subtle gridlines | Use high detail, mention "table may have faint lines" |

---

## Chart and Graph Analysis

### Extracting Data from Charts

```python
from pydantic import BaseModel

class DataPoint(BaseModel):
    label: str
    value: float
    unit: str | None

class ChartData(BaseModel):
    chart_type: str  # bar, line, pie, scatter
    title: str
    x_axis_label: str | None
    y_axis_label: str | None
    data_points: list[DataPoint]
    trend_description: str | None

prompt = """
Analyze this chart and extract:
1. Chart type
2. Title and axis labels
3. All data points with their values (estimate if not labeled)
4. Overall trend or key insight
"""

response = client.responses.parse(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": chart_url, "detail": "high"},
            {"type": "input_text", "text": prompt}
        ]
    }],
    text_format=ChartData
)

chart = response.output_parsed
print(f"{chart.chart_type}: {chart.title}")
for point in chart.data_points:
    print(f"  {point.label}: {point.value} {point.unit or ''}")
```

### Comparing Multiple Charts

```python
prompt = """
Compare these two charts showing quarterly revenue:

Chart 1 shows 2023 data.
Chart 2 shows 2024 data.

Analyze:
1. Year-over-year growth for each quarter
2. Which quarter improved most
3. Any concerning trends
"""

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": chart_2023, "detail": "high"},
            {"type": "input_text", "text": "[Chart 1: 2023]"},
            {"type": "input_image", "image_url": chart_2024, "detail": "high"},
            {"type": "input_text", "text": "[Chart 2: 2024]"}
        ]
    }]
)
```

---

## Form Field Detection

### Invoice Processing

```python
from pydantic import BaseModel
from datetime import date

class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total: float

class Address(BaseModel):
    street: str
    city: str
    state: str
    postal_code: str
    country: str

class Invoice(BaseModel):
    invoice_number: str
    invoice_date: date
    due_date: date | None
    vendor_name: str
    vendor_address: Address
    bill_to: str
    bill_to_address: Address | None
    line_items: list[LineItem]
    subtotal: float
    tax: float | None
    total: float
    payment_terms: str | None
    purchase_order: str | None

response = client.responses.parse(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{invoice_b64}",
                "detail": "high"
            },
            {
                "type": "input_text",
                "text": "Extract all invoice details. Use null for missing fields."
            }
        ]
    }],
    text_format=Invoice
)

invoice = response.output_parsed
print(f"Invoice #{invoice.invoice_number}")
print(f"Vendor: {invoice.vendor_name}")
print(f"Total: ${invoice.total}")
```

### Receipt Processing

```python
from pydantic import BaseModel

class ReceiptItem(BaseModel):
    name: str
    price: float
    quantity: int = 1

class Receipt(BaseModel):
    merchant_name: str
    merchant_address: str | None
    date: str
    time: str | None
    items: list[ReceiptItem]
    subtotal: float | None
    tax: float | None
    total: float
    payment_method: str | None
    last_four_digits: str | None

prompt = """
Extract receipt data. Common challenges:
- Faded text: estimate if partially visible
- Abbreviated items: expand if obvious
- Missing total: sum items + tax
"""

response = client.responses.parse(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": receipt_url, "detail": "high"},
            {"type": "input_text", "text": prompt}
        ]
    }],
    text_format=Receipt
)
```

### ID Document Extraction

```python
from pydantic import BaseModel

class DriverLicense(BaseModel):
    full_name: str
    date_of_birth: str
    license_number: str
    expiration_date: str
    address: str
    state: str
    class_type: str | None
    restrictions: list[str] | None

class Passport(BaseModel):
    surname: str
    given_names: str
    nationality: str
    date_of_birth: str
    passport_number: str
    expiration_date: str
    sex: str
    place_of_birth: str | None

# Use with appropriate security measures
response = client.responses.parse(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": document_url, "detail": "high"},
            {
                "type": "input_text",
                "text": "Extract fields from this ID document."
            }
        ]
    }],
    text_format=DriverLicense
)
```

> **⚠️ Warning:** Handle PII responsibly. Don't log or store document images longer than necessary. Use secure transmission and comply with privacy regulations.

---

## OCR from Imperfect Images

### Handling Low Quality Scans

```python
prompt = """
Extract text from this scanned document.

Document quality notes:
- May have faded or smudged text
- Slight skew is possible
- Some words may be partially visible

Instructions:
1. Extract all readable text
2. Mark uncertain words with [?]
3. Note any completely illegible sections
4. Preserve paragraph structure
"""

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": scan_url, "detail": "high"},
            {"type": "input_text", "text": prompt}
        ]
    }]
)
```

### Handwritten Text

```python
prompt = """
Transcribe this handwritten note.

Guidelines:
1. Do your best to interpret cursive writing
2. Mark unclear words with [unclear: best_guess]
3. Preserve line breaks as in original
4. Note if any sections are completely illegible
"""

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": handwritten_url, "detail": "high"},
            {"type": "input_text", "text": prompt}
        ]
    }]
)
```

### Multi-Language Documents

```python
prompt = """
This document contains text in multiple languages.

Extract all text, organizing by language:
1. Identify each language present
2. Group text by language
3. Provide translation to English for non-English text

Note: Document may contain Chinese, Japanese, or Korean characters.
Some models have limited accuracy with non-Latin scripts.
"""

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": multilang_doc, "detail": "high"},
            {"type": "input_text", "text": prompt}
        ]
    }]
)
```

---

## Document Understanding Pipeline

### Complete Document Processing System

```python
from pydantic import BaseModel
from enum import Enum

class DocumentType(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    FORM = "form"
    REPORT = "report"
    UNKNOWN = "unknown"

class DocumentClassification(BaseModel):
    document_type: DocumentType
    confidence: float
    language: str
    page_count: int
    contains_tables: bool
    contains_charts: bool
    contains_signatures: bool

def classify_document(image_url: str) -> DocumentClassification:
    """First step: classify document type."""
    
    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": image_url, "detail": "low"},
                {
                    "type": "input_text",
                    "text": "Classify this document. What type is it? What features does it contain?"
                }
            ]
        }],
        text_format=DocumentClassification
    )
    
    return response.output_parsed

def process_document(image_url: str):
    """Complete document processing pipeline."""
    
    # Step 1: Classify
    classification = classify_document(image_url)
    print(f"Document type: {classification.document_type}")
    
    # Step 2: Route to appropriate extractor
    if classification.document_type == DocumentType.INVOICE:
        return extract_invoice(image_url)
    elif classification.document_type == DocumentType.RECEIPT:
        return extract_receipt(image_url)
    elif classification.document_type == DocumentType.CONTRACT:
        return extract_contract(image_url)
    else:
        return extract_generic(image_url)

# Usage
result = process_document("https://example.com/document.pdf")
```

---

## Common Mistakes

### ❌ Low Detail for Text-Heavy Documents

```python
# Bad: Can't read small text
{"detail": "low"}

# Good: Full resolution
{"detail": "high"}
```

### ❌ No Structure in Extraction

```python
# Bad: Unstructured extraction
prompt = "Get info from this invoice"

# Good: Specific fields
prompt = """
Extract these specific fields:
- Invoice number (format: INV-XXXXX)
- Invoice date (YYYY-MM-DD)
- Due date
- Line items: description, quantity, unit price, total
- Grand total
"""
```

### ❌ Ignoring Document Quality

```python
# Bad: Assuming perfect scans
prompt = "Extract all text"

# Good: Accounting for quality issues
prompt = """
Extract text from this scanned document.
Quality considerations:
- Some text may be faded
- Document might be slightly skewed
- Mark uncertain readings with [?]
"""
```

---

## Hands-on Exercise

### Your Task

Build an expense report processor that:
1. Accepts a receipt image
2. Extracts merchant, date, items, and total
3. Categorizes the expense (meals, transport, supplies, etc.)
4. Returns structured data ready for accounting software

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
from datetime import date

class ExpenseCategory(str, Enum):
    MEALS = "meals"
    TRANSPORT = "transport"
    SUPPLIES = "supplies"
    LODGING = "lodging"
    ENTERTAINMENT = "entertainment"
    OTHER = "other"

class ExpenseItem(BaseModel):
    description: str
    amount: float
    category_override: ExpenseCategory | None = None

class ProcessedExpense(BaseModel):
    merchant_name: str
    merchant_category: ExpenseCategory
    expense_date: date
    items: list[ExpenseItem]
    subtotal: float
    tax: float | None
    total: float
    currency: str
    payment_method: str | None
    receipt_quality_score: int  # 1-10
    extraction_notes: list[str]

def process_expense_receipt(image_url: str) -> ProcessedExpense:
    """Process receipt and return structured expense data."""
    
    client = OpenAI()
    
    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": image_url,
                    "detail": "high"
                },
                {
                    "type": "input_text",
                    "text": """
                    Process this receipt for expense reporting.
                    
                    Categorize the merchant as:
                    - meals: restaurants, cafes, food delivery
                    - transport: uber, taxi, gas, parking
                    - supplies: office supplies, hardware
                    - lodging: hotels, airbnb
                    - entertainment: events, tickets
                    - other: anything else
                    
                    Rate receipt quality 1-10 (10 = perfect clarity).
                    Note any extraction difficulties.
                    """
                }
            ]
        }],
        text_format=ProcessedExpense
    )
    
    return response.output_parsed

# Usage
expense = process_expense_receipt("https://example.com/receipt.jpg")

print(f"Merchant: {expense.merchant_name}")
print(f"Category: {expense.merchant_category}")
print(f"Date: {expense.expense_date}")
print(f"Total: {expense.currency} {expense.total}")
print(f"Quality Score: {expense.receipt_quality_score}/10")

if expense.extraction_notes:
    print("Notes:", ", ".join(expense.extraction_notes))
```

</details>

---

## Summary

✅ **Gemini has native PDF support:** Up to 1,000 pages, 258 tokens/page
✅ **OpenAI/Anthropic need images:** Convert PDF pages for processing
✅ **Tables need high detail:** Use `detail: "high"` for text extraction
✅ **Structured output helps:** Define Pydantic models for consistent extraction
✅ **Handle quality issues:** Account for scans, fading, handwriting in prompts

**Next:** [Vision Capabilities](./03-vision-capabilities.md)

---

## Further Reading

- [Gemini PDF Processing](https://ai.google.dev/gemini-api/docs/document-processing)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [Building Document AI](https://cloud.google.com/document-ai/docs/overview)

---

<!-- 
Sources Consulted:
- Gemini Document Processing: Native PDF, 1000 pages, 50MB limit
- OpenAI Vision Guide: Image processing workflow
- Anthropic Vision: 100 image limit, base64 format
-->
