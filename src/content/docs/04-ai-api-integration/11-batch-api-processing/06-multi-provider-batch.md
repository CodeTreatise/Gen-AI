---
title: "Multi-Provider Batch APIs"
---

# Multi-Provider Batch APIs

## Introduction

While OpenAI pioneered batch APIs for LLMs, Anthropic and Google now offer similar capabilities. This lesson covers cross-provider batch processing including Claude Message Batches API, Gemini Batch API, and batch processing with structured outputs.

### What We'll Cover

- Anthropic Message Batches API
- Google Gemini Batch API
- Cross-provider abstraction
- Batch with structured outputs
- Provider selection strategies

### Prerequisites

- OpenAI Batch API fundamentals
- Multi-provider API integration
- JSON Schema / Pydantic basics

---

## Provider Comparison

```mermaid
flowchart TD
    subgraph OpenAI["OpenAI Batch"]
        OI[File Upload] --> OB[Create Batch]
        OB --> OP[Poll Status]
        OP --> OD[Download Results]
    end
    
    subgraph Anthropic["Anthropic Batch"]
        AI[Direct Request] --> AB[Create Batch]
        AB --> AP[Poll/Webhook]
        AP --> AR[Results in Response]
    end
    
    subgraph Google["Google Batch"]
        GI[GCS Upload] --> GB[Create Job]
        GB --> GP[Poll Status]
        GP --> GD[Download from GCS]
    end
```

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| Input Method | JSONL File Upload | JSON Array in Request | GCS File |
| Max Requests | 50,000 per batch | 10,000 per batch | 30,000 per batch |
| Discount | 50% | 50% | 50% |
| Completion Window | 24 hours | 24 hours | Variable |
| Result Retrieval | Download File | In API Response | GCS Download |
| Webhook Support | Limited | Yes | Cloud Pub/Sub |

---

## Anthropic Message Batches API

```python
from anthropic import Anthropic
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import time

client = Anthropic()

class BatchStatus(Enum):
    """Anthropic batch status values."""
    
    IN_PROGRESS = "in_progress"
    ENDED = "ended"
    CANCELING = "canceling"


@dataclass
class MessageBatchRequest:
    """A single request in an Anthropic batch."""
    
    custom_id: str
    model: str
    max_tokens: int
    messages: List[Dict[str, str]]
    system: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to API format."""
        
        params = {
            "custom_id": self.custom_id,
            "params": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": self.messages
            }
        }
        
        if self.system:
            params["params"]["system"] = self.system
        
        return params


class AnthropicBatchProcessor:
    """Process batches with Anthropic API."""
    
    def __init__(self):
        self.client = Anthropic()
    
    def create_batch(
        self,
        requests: List[MessageBatchRequest]
    ) -> dict:
        """Create a message batch."""
        
        # Convert requests to API format
        batch_requests = [r.to_dict() for r in requests]
        
        # Create batch
        batch = self.client.messages.batches.create(
            requests=batch_requests
        )
        
        print(f"Created batch: {batch.id}")
        print(f"Status: {batch.processing_status}")
        print(f"Requests: {batch.request_counts.processing}")
        
        return {
            "id": batch.id,
            "status": batch.processing_status,
            "created_at": batch.created_at.isoformat()
        }
    
    def get_status(self, batch_id: str) -> dict:
        """Get batch status."""
        
        batch = self.client.messages.batches.retrieve(batch_id)
        
        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
        completed = counts.succeeded + counts.errored
        
        return {
            "id": batch.id,
            "status": batch.processing_status,
            "total": total,
            "succeeded": counts.succeeded,
            "errored": counts.errored,
            "canceled": counts.canceled,
            "expired": counts.expired,
            "processing": counts.processing,
            "progress": completed / total if total > 0 else 0,
            "ended_at": batch.ended_at.isoformat() if batch.ended_at else None
        }
    
    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: float = 30.0,
        timeout: float = 86400.0
    ) -> dict:
        """Poll until batch completes."""
        
        start = time.time()
        
        while True:
            status = self.get_status(batch_id)
            
            print(
                f"Status: {status['status']} - "
                f"{status['progress']:.1%} complete"
            )
            
            if status["status"] == "ended":
                return status
            
            if time.time() - start > timeout:
                raise TimeoutError(f"Batch {batch_id} timed out")
            
            time.sleep(poll_interval)
    
    def get_results(self, batch_id: str) -> List[dict]:
        """Retrieve batch results."""
        
        results = []
        
        # Results are streamed
        for result in self.client.messages.batches.results(batch_id):
            results.append({
                "custom_id": result.custom_id,
                "type": result.result.type,  # "succeeded", "errored", etc.
                "message": result.result.message if result.result.type == "succeeded" else None,
                "error": result.result.error if result.result.type == "errored" else None
            })
        
        return results
    
    def process_batch(
        self,
        requests: List[MessageBatchRequest]
    ) -> List[dict]:
        """Complete batch processing workflow."""
        
        # Create batch
        batch_info = self.create_batch(requests)
        
        # Wait for completion
        final_status = self.wait_for_completion(batch_info["id"])
        
        print(f"Batch completed: {final_status['succeeded']} succeeded, "
              f"{final_status['errored']} errored")
        
        # Get results
        return self.get_results(batch_info["id"])


# Usage
processor = AnthropicBatchProcessor()

# Create requests
requests = [
    MessageBatchRequest(
        custom_id=f"req-{i}",
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": f"Summarize this: {doc}"}],
        system="You are a helpful summarizer."
    )
    for i, doc in enumerate(documents)
]

# Process
results = processor.process_batch(requests)

# Handle results
for result in results:
    if result["type"] == "succeeded":
        content = result["message"].content[0].text
        print(f"{result['custom_id']}: {content[:100]}...")
    else:
        print(f"{result['custom_id']}: Error - {result['error']}")
```

---

## Anthropic Batch Lifecycle

```python
@dataclass
class AnthropicBatchResult:
    """Result from Anthropic batch."""
    
    custom_id: str
    result_type: str  # succeeded, errored, canceled, expired
    content: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    error: Optional[dict] = None
    stop_reason: Optional[str] = None
    
    @classmethod
    def from_api(cls, result) -> 'AnthropicBatchResult':
        """Create from API response."""
        
        if result.result.type == "succeeded":
            message = result.result.message
            content_blocks = message.content
            content = "".join(
                block.text for block in content_blocks 
                if hasattr(block, 'text')
            )
            
            return cls(
                custom_id=result.custom_id,
                result_type="succeeded",
                content=content,
                usage={
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens
                },
                stop_reason=message.stop_reason
            )
        else:
            return cls(
                custom_id=result.custom_id,
                result_type=result.result.type,
                error=result.result.error.dict() if result.result.error else None
            )


class AnthropicBatchManager:
    """Complete batch management for Anthropic."""
    
    def __init__(self):
        self.client = Anthropic()
        self.active_batches: Dict[str, dict] = {}
    
    def list_batches(self, limit: int = 20) -> List[dict]:
        """List recent batches."""
        
        batches = self.client.messages.batches.list(limit=limit)
        
        return [
            {
                "id": b.id,
                "status": b.processing_status,
                "created_at": b.created_at.isoformat(),
                "request_counts": {
                    "processing": b.request_counts.processing,
                    "succeeded": b.request_counts.succeeded,
                    "errored": b.request_counts.errored
                }
            }
            for b in batches.data
        ]
    
    def cancel_batch(self, batch_id: str) -> dict:
        """Cancel a running batch."""
        
        batch = self.client.messages.batches.cancel(batch_id)
        
        return {
            "id": batch.id,
            "status": batch.processing_status,
            "message": "Cancel initiated - partial results may be available"
        }
    
    def get_results_by_status(
        self,
        batch_id: str
    ) -> Dict[str, List[AnthropicBatchResult]]:
        """Get results grouped by status."""
        
        results = {
            "succeeded": [],
            "errored": [],
            "canceled": [],
            "expired": []
        }
        
        for result in self.client.messages.batches.results(batch_id):
            parsed = AnthropicBatchResult.from_api(result)
            
            if parsed.result_type in results:
                results[parsed.result_type].append(parsed)
        
        return results
    
    def calculate_cost(
        self,
        results: List[AnthropicBatchResult],
        model: str = "claude-sonnet-4"
    ) -> dict:
        """Calculate batch cost."""
        
        # Batch pricing (50% discount)
        pricing = {
            "claude-sonnet-4": (0.0015, 0.0075),  # per 1K tokens
            "claude-haiku": (0.0004, 0.002)
        }
        
        input_price, output_price = pricing.get(model, (0.003, 0.015))
        
        total_input = sum(r.usage.get("input_tokens", 0) for r in results if r.usage)
        total_output = sum(r.usage.get("output_tokens", 0) for r in results if r.usage)
        
        input_cost = (total_input / 1000) * input_price
        output_cost = (total_output / 1000) * output_price
        
        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "savings_vs_realtime": input_cost + output_cost  # Same as cost (50% discount)
        }
```

---

## Google Gemini Batch API

```python
from google.cloud import aiplatform
from google.cloud import storage
from dataclasses import dataclass
from typing import List, Optional
import json
import time

@dataclass
class GeminiBatchRequest:
    """Request for Gemini batch."""
    
    request_id: str
    contents: List[dict]
    generation_config: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to API format."""
        
        request = {
            "request": {
                "contents": self.contents
            }
        }
        
        if self.generation_config:
            request["request"]["generationConfig"] = self.generation_config
        
        return request


class GeminiBatchProcessor:
    """Process batches with Gemini API."""
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        bucket_name: str = "my-batch-bucket"
    ):
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        
        # Initialize clients
        aiplatform.init(project=project_id, location=location)
        self.storage_client = storage.Client()
    
    def upload_input(
        self,
        requests: List[GeminiBatchRequest],
        input_path: str = "batch_input.jsonl"
    ) -> str:
        """Upload input file to GCS."""
        
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(input_path)
        
        # Create JSONL content
        lines = [json.dumps(r.to_dict()) for r in requests]
        content = '\n'.join(lines)
        
        blob.upload_from_string(content)
        
        gcs_uri = f"gs://{self.bucket_name}/{input_path}"
        print(f"Uploaded input to: {gcs_uri}")
        
        return gcs_uri
    
    def create_batch_job(
        self,
        input_uri: str,
        output_uri: str,
        model: str = "gemini-2.0-flash"
    ) -> str:
        """Create a batch prediction job."""
        
        # Full model resource name
        model_name = f"publishers/google/models/{model}"
        
        # Create job
        job = aiplatform.BatchPredictionJob.create(
            job_display_name=f"batch-{int(time.time())}",
            model_name=model_name,
            gcs_source=input_uri,
            gcs_destination_prefix=output_uri,
            sync=False  # Don't wait
        )
        
        print(f"Created job: {job.name}")
        return job.name
    
    def get_job_status(self, job_name: str) -> dict:
        """Get job status."""
        
        job = aiplatform.BatchPredictionJob(job_name)
        
        return {
            "name": job.name,
            "state": job.state.name,
            "create_time": job.create_time.isoformat() if job.create_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None,
            "output_location": job.output_info.gcs_output_directory if job.output_info else None
        }
    
    def wait_for_completion(
        self,
        job_name: str,
        poll_interval: float = 60.0,
        timeout: float = 86400.0
    ) -> dict:
        """Wait for job completion."""
        
        job = aiplatform.BatchPredictionJob(job_name)
        
        start = time.time()
        
        while True:
            job._sync_gca_resource()  # Refresh status
            
            state = job.state.name
            print(f"Status: {state}")
            
            if state in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                return self.get_job_status(job_name)
            
            if time.time() - start > timeout:
                raise TimeoutError(f"Job {job_name} timed out")
            
            time.sleep(poll_interval)
    
    def download_results(self, output_uri: str) -> List[dict]:
        """Download and parse results from GCS."""
        
        # Parse GCS URI
        # gs://bucket/path/to/output
        parts = output_uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        
        bucket = self.storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        results = []
        
        for blob in blobs:
            if blob.name.endswith('.jsonl'):
                content = blob.download_as_text()
                
                for line in content.strip().split('\n'):
                    if line:
                        results.append(json.loads(line))
        
        return results
    
    def process_batch(
        self,
        requests: List[GeminiBatchRequest],
        model: str = "gemini-2.0-flash"
    ) -> List[dict]:
        """Complete batch processing workflow."""
        
        timestamp = int(time.time())
        input_uri = self.upload_input(
            requests,
            f"input/batch_{timestamp}.jsonl"
        )
        
        output_uri = f"gs://{self.bucket_name}/output/batch_{timestamp}/"
        
        # Create job
        job_name = self.create_batch_job(input_uri, output_uri, model)
        
        # Wait
        final_status = self.wait_for_completion(job_name)
        
        if final_status["state"] != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Job failed: {final_status['state']}")
        
        # Download results
        return self.download_results(final_status["output_location"])


# Usage (requires GCP setup)
"""
processor = GeminiBatchProcessor(
    project_id="my-project",
    bucket_name="my-batch-bucket"
)

requests = [
    GeminiBatchRequest(
        request_id=f"req-{i}",
        contents=[{"role": "user", "parts": [{"text": f"Analyze: {text}"}]}],
        generation_config={"temperature": 0.7}
    )
    for i, text in enumerate(texts)
]

results = processor.process_batch(requests)
"""
```

---

## Cross-Provider Abstraction

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum

class Provider(Enum):
    """Supported batch providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class UnifiedBatchRequest:
    """Provider-agnostic batch request."""
    
    custom_id: str
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    system: Optional[str] = None


@dataclass
class UnifiedBatchResult:
    """Provider-agnostic batch result."""
    
    custom_id: str
    success: bool
    content: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    provider: Optional[Provider] = None


class BatchProvider(ABC):
    """Abstract batch provider."""
    
    @abstractmethod
    def submit(self, requests: List[UnifiedBatchRequest]) -> str:
        """Submit batch, return batch ID."""
        pass
    
    @abstractmethod
    def status(self, batch_id: str) -> dict:
        """Get batch status."""
        pass
    
    @abstractmethod
    def results(self, batch_id: str) -> List[UnifiedBatchResult]:
        """Get batch results."""
        pass
    
    @abstractmethod
    def cancel(self, batch_id: str) -> bool:
        """Cancel batch."""
        pass


class OpenAIBatchProvider(BatchProvider):
    """OpenAI implementation."""
    
    def __init__(self, default_model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI()
        self.default_model = default_model
    
    def _convert_request(self, req: UnifiedBatchRequest) -> dict:
        """Convert to OpenAI format."""
        
        messages = req.messages.copy()
        if req.system:
            messages.insert(0, {"role": "system", "content": req.system})
        
        return {
            "custom_id": req.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": req.model or self.default_model,
                "messages": messages,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature
            }
        }
    
    def submit(self, requests: List[UnifiedBatchRequest]) -> str:
        """Submit batch to OpenAI."""
        
        import json
        import tempfile
        
        # Create JSONL
        lines = [json.dumps(self._convert_request(r)) for r in requests]
        content = '\n'.join(lines)
        
        # Upload file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(content)
            f.flush()
            
            with open(f.name, 'rb') as upload_file:
                file = self.client.files.create(
                    file=upload_file,
                    purpose="batch"
                )
        
        # Create batch
        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        return batch.id
    
    def status(self, batch_id: str) -> dict:
        batch = self.client.batches.retrieve(batch_id)
        return {
            "status": batch.status,
            "progress": batch.request_counts.completed / batch.request_counts.total
                       if batch.request_counts.total > 0 else 0
        }
    
    def results(self, batch_id: str) -> List[UnifiedBatchResult]:
        batch = self.client.batches.retrieve(batch_id)
        
        if not batch.output_file_id:
            return []
        
        content = self.client.files.content(batch.output_file_id).text
        results = []
        
        import json
        for line in content.strip().split('\n'):
            data = json.loads(line)
            
            if data.get("response", {}).get("status_code") == 200:
                body = data["response"]["body"]
                results.append(UnifiedBatchResult(
                    custom_id=data["custom_id"],
                    success=True,
                    content=body["choices"][0]["message"]["content"],
                    usage=body.get("usage"),
                    provider=Provider.OPENAI
                ))
            else:
                results.append(UnifiedBatchResult(
                    custom_id=data["custom_id"],
                    success=False,
                    error=str(data.get("error")),
                    provider=Provider.OPENAI
                ))
        
        return results
    
    def cancel(self, batch_id: str) -> bool:
        self.client.batches.cancel(batch_id)
        return True


class AnthropicBatchProvider(BatchProvider):
    """Anthropic implementation."""
    
    def __init__(self, default_model: str = "claude-sonnet-4-20250514"):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.default_model = default_model
    
    def _convert_request(self, req: UnifiedBatchRequest) -> dict:
        """Convert to Anthropic format."""
        
        params = {
            "model": req.model or self.default_model,
            "max_tokens": req.max_tokens,
            "messages": req.messages
        }
        
        if req.system:
            params["system"] = req.system
        
        return {
            "custom_id": req.custom_id,
            "params": params
        }
    
    def submit(self, requests: List[UnifiedBatchRequest]) -> str:
        batch_requests = [self._convert_request(r) for r in requests]
        
        batch = self.client.messages.batches.create(
            requests=batch_requests
        )
        
        return batch.id
    
    def status(self, batch_id: str) -> dict:
        batch = self.client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored
        
        return {
            "status": batch.processing_status,
            "progress": (counts.succeeded + counts.errored) / total if total > 0 else 0
        }
    
    def results(self, batch_id: str) -> List[UnifiedBatchResult]:
        results = []
        
        for result in self.client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                message = result.result.message
                content = "".join(
                    b.text for b in message.content if hasattr(b, 'text')
                )
                
                results.append(UnifiedBatchResult(
                    custom_id=result.custom_id,
                    success=True,
                    content=content,
                    usage={
                        "input_tokens": message.usage.input_tokens,
                        "output_tokens": message.usage.output_tokens
                    },
                    provider=Provider.ANTHROPIC
                ))
            else:
                results.append(UnifiedBatchResult(
                    custom_id=result.custom_id,
                    success=False,
                    error=str(result.result.error),
                    provider=Provider.ANTHROPIC
                ))
        
        return results
    
    def cancel(self, batch_id: str) -> bool:
        self.client.messages.batches.cancel(batch_id)
        return True


class UnifiedBatchProcessor:
    """Unified batch processor across providers."""
    
    def __init__(self):
        self.providers = {
            Provider.OPENAI: OpenAIBatchProvider(),
            Provider.ANTHROPIC: AnthropicBatchProvider()
        }
    
    def process(
        self,
        provider: Provider,
        requests: List[UnifiedBatchRequest],
        poll_interval: float = 30.0
    ) -> List[UnifiedBatchResult]:
        """Process batch with specified provider."""
        
        impl = self.providers.get(provider)
        if not impl:
            raise ValueError(f"Provider not configured: {provider}")
        
        # Submit
        batch_id = impl.submit(requests)
        print(f"Submitted to {provider.value}: {batch_id}")
        
        # Poll
        import time
        while True:
            status = impl.status(batch_id)
            print(f"Status: {status['status']} ({status['progress']:.1%})")
            
            if status["status"] in ["completed", "ended", "failed", "cancelled"]:
                break
            
            time.sleep(poll_interval)
        
        # Get results
        return impl.results(batch_id)


# Usage
processor = UnifiedBatchProcessor()

requests = [
    UnifiedBatchRequest(
        custom_id=f"doc-{i}",
        messages=[{"role": "user", "content": f"Summarize: {doc}"}],
        system="You are a helpful summarizer.",
        max_tokens=500
    )
    for i, doc in enumerate(documents)
]

# Use any provider with same interface
results = processor.process(Provider.ANTHROPIC, requests)

for result in results:
    if result.success:
        print(f"{result.custom_id}: {result.content[:100]}...")
```

---

## Batch with Structured Outputs

```python
from pydantic import BaseModel, Field
from typing import List
import json

# Define output schema
class DocumentAnalysis(BaseModel):
    """Structured output for document analysis."""
    
    title: str = Field(description="Document title")
    summary: str = Field(description="Brief summary")
    key_points: List[str] = Field(description="Main points")
    sentiment: str = Field(description="Overall sentiment: positive, negative, neutral")
    confidence: float = Field(ge=0, le=1, description="Confidence score")


class StructuredBatchProcessor:
    """Batch processing with structured outputs."""
    
    def __init__(self, provider: Provider = Provider.OPENAI):
        self.provider = provider
        
        if provider == Provider.OPENAI:
            from openai import OpenAI
            self.client = OpenAI()
    
    def create_structured_request(
        self,
        custom_id: str,
        content: str,
        schema: type[BaseModel]
    ) -> dict:
        """Create request with structured output schema."""
        
        # OpenAI format with response_format
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-2024-08-06",
                "messages": [
                    {
                        "role": "system",
                        "content": "Analyze the document and return structured data."
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": schema.model_json_schema()
                    }
                }
            }
        }
    
    def submit_structured_batch(
        self,
        items: List[tuple],  # (custom_id, content)
        schema: type[BaseModel]
    ) -> str:
        """Submit batch with structured output schema."""
        
        import tempfile
        
        # Create requests
        lines = []
        for custom_id, content in items:
            request = self.create_structured_request(custom_id, content, schema)
            lines.append(json.dumps(request))
        
        content = '\n'.join(lines)
        
        # Upload and create batch
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(content)
            f.flush()
            
            with open(f.name, 'rb') as upload_file:
                file = self.client.files.create(
                    file=upload_file,
                    purpose="batch"
                )
        
        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        return batch.id
    
    def parse_structured_results(
        self,
        batch_id: str,
        schema: type[BaseModel]
    ) -> Dict[str, BaseModel]:
        """Parse results into structured objects."""
        
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.status != "completed" or not batch.output_file_id:
            raise RuntimeError(f"Batch not ready: {batch.status}")
        
        content = self.client.files.content(batch.output_file_id).text
        results = {}
        
        for line in content.strip().split('\n'):
            data = json.loads(line)
            custom_id = data["custom_id"]
            
            if data.get("response", {}).get("status_code") == 200:
                message_content = data["response"]["body"]["choices"][0]["message"]["content"]
                
                # Parse JSON and validate with schema
                parsed = schema.model_validate_json(message_content)
                results[custom_id] = parsed
        
        return results


# Usage
processor = StructuredBatchProcessor()

# Prepare items
items = [
    (f"doc-{i}", document)
    for i, document in enumerate(documents)
]

# Submit with schema
batch_id = processor.submit_structured_batch(items, DocumentAnalysis)
print(f"Submitted batch: {batch_id}")

# After completion, get typed results
results = processor.parse_structured_results(batch_id, DocumentAnalysis)

for custom_id, analysis in results.items():
    print(f"{custom_id}:")
    print(f"  Title: {analysis.title}")
    print(f"  Sentiment: {analysis.sentiment}")
    print(f"  Confidence: {analysis.confidence}")
    print(f"  Key points: {len(analysis.key_points)}")
```

---

## Provider Selection Strategy

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class BatchRequirements:
    """Requirements for batch processing."""
    
    request_count: int
    avg_input_tokens: int
    avg_output_tokens: int
    max_latency_hours: float = 24.0
    requires_structured_output: bool = False
    requires_vision: bool = False
    budget_per_request: Optional[float] = None


class ProviderSelector:
    """Select optimal provider for batch requirements."""
    
    PROVIDER_LIMITS = {
        Provider.OPENAI: {
            "max_requests": 50_000,
            "max_tokens_per_request": 128_000,
            "structured_output": True,
            "vision": True,
            "batch_discount": 0.5
        },
        Provider.ANTHROPIC: {
            "max_requests": 10_000,
            "max_tokens_per_request": 200_000,
            "structured_output": False,  # JSON mode, not strict
            "vision": True,
            "batch_discount": 0.5
        },
        Provider.GOOGLE: {
            "max_requests": 30_000,
            "max_tokens_per_request": 1_000_000,
            "structured_output": True,
            "vision": True,
            "batch_discount": 0.5
        }
    }
    
    def select(self, requirements: BatchRequirements) -> List[Provider]:
        """Select suitable providers for requirements."""
        
        suitable = []
        
        for provider, limits in self.PROVIDER_LIMITS.items():
            # Check request count
            if requirements.request_count > limits["max_requests"]:
                continue
            
            # Check token limits
            total_tokens = requirements.avg_input_tokens + requirements.avg_output_tokens
            if total_tokens > limits["max_tokens_per_request"]:
                continue
            
            # Check structured output
            if requirements.requires_structured_output and not limits["structured_output"]:
                continue
            
            # Check vision
            if requirements.requires_vision and not limits["vision"]:
                continue
            
            suitable.append(provider)
        
        return suitable
    
    def recommend(self, requirements: BatchRequirements) -> Optional[Provider]:
        """Recommend best provider for requirements."""
        
        suitable = self.select(requirements)
        
        if not suitable:
            return None
        
        # Priority: OpenAI for structured, Anthropic for long context, Google for volume
        if requirements.requires_structured_output:
            if Provider.OPENAI in suitable:
                return Provider.OPENAI
            if Provider.GOOGLE in suitable:
                return Provider.GOOGLE
        
        if requirements.avg_input_tokens > 50_000:
            if Provider.ANTHROPIC in suitable:
                return Provider.ANTHROPIC
            if Provider.GOOGLE in suitable:
                return Provider.GOOGLE
        
        if requirements.request_count > 20_000:
            if Provider.GOOGLE in suitable:
                return Provider.GOOGLE
            if Provider.OPENAI in suitable:
                return Provider.OPENAI
        
        # Default to OpenAI
        return suitable[0]


# Usage
selector = ProviderSelector()

requirements = BatchRequirements(
    request_count=5000,
    avg_input_tokens=1000,
    avg_output_tokens=500,
    requires_structured_output=True
)

recommended = selector.recommend(requirements)
print(f"Recommended provider: {recommended.value}")

# Get all suitable options
suitable = selector.select(requirements)
print(f"All suitable: {[p.value for p in suitable]}")
```

---

## Hands-on Exercise

### Your Task

Build a multi-provider batch failover system.

### Requirements

1. Try primary provider first
2. On failure, fall back to secondary provider
3. Merge results from multiple providers
4. Track which provider processed each request

### Expected Result

```python
failover = BatchFailover(
    primary=Provider.OPENAI,
    fallback=Provider.ANTHROPIC
)

results = failover.process(requests)
# Tries OpenAI first, falls back to Anthropic on failure
```

<details>
<summary>💡 Hints</summary>

- Handle partial failures from primary
- Retry only failed requests with fallback
- Combine results maintaining custom_id mapping
</details>

<details>
<summary>✅ Solution</summary>

```python
from typing import Set

class BatchFailover:
    """Multi-provider batch with automatic failover."""
    
    def __init__(
        self,
        primary: Provider,
        fallback: Provider,
        max_retries: int = 2
    ):
        self.primary = primary
        self.fallback = fallback
        self.max_retries = max_retries
        
        self.providers = {
            Provider.OPENAI: OpenAIBatchProvider(),
            Provider.ANTHROPIC: AnthropicBatchProvider()
        }
    
    def process(
        self,
        requests: List[UnifiedBatchRequest]
    ) -> Dict[str, UnifiedBatchResult]:
        """Process with failover."""
        
        all_results: Dict[str, UnifiedBatchResult] = {}
        pending_requests = requests.copy()
        
        # Try primary provider
        print(f"Trying primary: {self.primary.value}")
        
        try:
            primary_results = self._process_with_provider(
                self.primary, pending_requests
            )
            
            # Collect successes
            for result in primary_results:
                all_results[result.custom_id] = result
                if result.success:
                    # Remove from pending
                    pending_requests = [
                        r for r in pending_requests 
                        if r.custom_id != result.custom_id
                    ]
            
            print(f"Primary completed: {len(primary_results)} results, "
                  f"{len(pending_requests)} pending")
        
        except Exception as e:
            print(f"Primary failed: {e}")
            # All requests still pending
        
        # If any pending, try fallback
        if pending_requests:
            print(f"Trying fallback: {self.fallback.value}")
            
            try:
                fallback_results = self._process_with_provider(
                    self.fallback, pending_requests
                )
                
                for result in fallback_results:
                    all_results[result.custom_id] = result
                
                print(f"Fallback completed: {len(fallback_results)} results")
            
            except Exception as e:
                print(f"Fallback also failed: {e}")
                
                # Mark remaining as failed
                for req in pending_requests:
                    if req.custom_id not in all_results:
                        all_results[req.custom_id] = UnifiedBatchResult(
                            custom_id=req.custom_id,
                            success=False,
                            error=f"All providers failed: {e}"
                        )
        
        return all_results
    
    def _process_with_provider(
        self,
        provider: Provider,
        requests: List[UnifiedBatchRequest]
    ) -> List[UnifiedBatchResult]:
        """Process batch with specific provider."""
        
        impl = self.providers.get(provider)
        if not impl:
            raise ValueError(f"Provider not available: {provider}")
        
        # Submit
        batch_id = impl.submit(requests)
        
        # Poll
        import time
        while True:
            status = impl.status(batch_id)
            
            if status["status"] in ["completed", "ended"]:
                break
            
            if status["status"] in ["failed", "cancelled"]:
                raise RuntimeError(f"Batch failed: {status}")
            
            time.sleep(30)
        
        # Get results
        return impl.results(batch_id)
    
    def get_provider_stats(
        self,
        results: Dict[str, UnifiedBatchResult]
    ) -> dict:
        """Get statistics by provider."""
        
        stats = {
            "total": len(results),
            "by_provider": {},
            "success_rate": 0
        }
        
        successes = 0
        
        for result in results.values():
            provider_name = result.provider.value if result.provider else "unknown"
            
            if provider_name not in stats["by_provider"]:
                stats["by_provider"][provider_name] = {
                    "total": 0,
                    "success": 0,
                    "failed": 0
                }
            
            stats["by_provider"][provider_name]["total"] += 1
            
            if result.success:
                stats["by_provider"][provider_name]["success"] += 1
                successes += 1
            else:
                stats["by_provider"][provider_name]["failed"] += 1
        
        stats["success_rate"] = successes / len(results) if results else 0
        
        return stats


# Usage
failover = BatchFailover(
    primary=Provider.OPENAI,
    fallback=Provider.ANTHROPIC
)

requests = [
    UnifiedBatchRequest(
        custom_id=f"doc-{i}",
        messages=[{"role": "user", "content": f"Analyze: {doc}"}],
        max_tokens=500
    )
    for i, doc in enumerate(documents)
]

results = failover.process(requests)

# Show stats
stats = failover.get_provider_stats(results)
print(f"Total: {stats['total']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"By provider: {stats['by_provider']}")
```

</details>

---

## Summary

✅ Anthropic Message Batches use direct API requests, not file uploads  
✅ Google Gemini Batch requires GCS for input/output  
✅ Unified abstraction enables provider-agnostic code  
✅ Structured outputs work with batch processing  
✅ Provider selection considers limits, features, and cost

**Next:** [Webhook Patterns for AI](../12-webhook-patterns-for-ai.md)

---

## Further Reading

- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/message-batches) — Official docs
- [Google Batch Predictions](https://cloud.google.com/vertex-ai/docs/generative-ai/batch-prediction) — Vertex AI batch
- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — OpenAI reference

<!-- 
Sources Consulted:
- Anthropic Message Batches: https://docs.anthropic.com/en/docs/build-with-claude/message-batches
- OpenAI Batch API: https://platform.openai.com/docs/guides/batch
- Google Vertex AI: https://cloud.google.com/vertex-ai/docs/generative-ai/batch-prediction
-->
