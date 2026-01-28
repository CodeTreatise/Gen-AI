---
title: "Practical AI Applications"
---

# Practical AI Applications

## Introduction

This lesson applies concurrency patterns to real AI workflows: parallel data preprocessing, concurrent API requests, batch inference, and streaming processors.

### What We'll Cover

- Parallel data preprocessing
- Concurrent LLM API calls
- Batch model inference
- Background task workers
- Streaming processors

### Prerequisites

- All concurrency approaches
- Basic ML/AI concepts

---

## Parallel Data Preprocessing

### Image Processing Pipeline

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

def process_image(image_path: str) -> dict:
    """CPU-bound image processing."""
    # Simulate: load, resize, normalize, augment
    time.sleep(0.1)  # Simulates actual processing
    return {
        "path": image_path,
        "processed": True,
        "size": (224, 224)
    }

def process_images_parallel(image_paths: list[str]) -> list[dict]:
    """Process images in parallel across CPU cores."""
    n_workers = mp.cpu_count()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_image, image_paths))
    
    return results

if __name__ == '__main__':
    images = [f"image_{i}.jpg" for i in range(100)]
    
    start = time.perf_counter()
    results = process_images_parallel(images)
    elapsed = time.perf_counter() - start
    
    print(f"Processed {len(results)} images in {elapsed:.2f}s")
```

### Text Preprocessing Pipeline

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import re

def preprocess_text(text: str) -> dict:
    """CPU-bound text preprocessing."""
    # Clean
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize (simple)
    tokens = text.split()
    
    return {
        "original_length": len(text),
        "token_count": len(tokens),
        "tokens": tokens[:10]  # First 10
    }

def preprocess_batch(texts: list[str]) -> list[dict]:
    """Process texts in parallel."""
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        return list(executor.map(preprocess_text, texts))

if __name__ == '__main__':
    texts = ["This is sample text #{}!".format(i) for i in range(1000)]
    results = preprocess_batch(texts)
    print(f"Processed {len(results)} texts")
```

---

## Concurrent LLM API Calls

### Rate-Limited Batch Queries

```python
import asyncio
import httpx
from typing import AsyncIterator

class LLMClient:
    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.base_url = "https://api.openai.com/v1"
    
    async def complete(self, prompt: str) -> str:
        """Single completion with rate limiting."""
        async with self.semaphore:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
                data = response.json()
                return data["choices"][0]["message"]["content"]
    
    async def batch_complete(self, prompts: list[str]) -> list[str]:
        """Complete multiple prompts concurrently."""
        tasks = [self.complete(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Usage
async def main():
    client = LLMClient("your-api-key", max_concurrent=5)
    
    prompts = [
        "Summarize quantum computing",
        "Explain machine learning",
        "What is deep learning?"
    ]
    
    results = await client.batch_complete(prompts)
    for prompt, result in zip(prompts, results):
        if isinstance(result, Exception):
            print(f"Error: {result}")
        else:
            print(f"Q: {prompt[:30]}... A: {result[:50]}...")

# asyncio.run(main())
```

### Streaming Multiple Responses

```python
import asyncio
import httpx
from typing import AsyncIterator

async def stream_llm(
    prompt: str, 
    client: httpx.AsyncClient
) -> AsyncIterator[str]:
    """Stream response from LLM."""
    async with client.stream(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                import json
                chunk = json.loads(line[6:])
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    yield content

async def stream_multiple(prompts: list[str]):
    """Stream multiple prompts concurrently."""
    async with httpx.AsyncClient(timeout=60) as client:
        async def collect_stream(prompt: str, index: int):
            chunks = []
            async for chunk in stream_llm(prompt, client):
                chunks.append(chunk)
                print(f"[{index}] Chunk received")
            return "".join(chunks)
        
        tasks = [
            collect_stream(prompt, i) 
            for i, prompt in enumerate(prompts)
        ]
        return await asyncio.gather(*tasks)
```

---

## Batch Model Inference

### Parallel Inference with ProcessPool

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import time

# Simulate a model (in practice, use sklearn, torch, etc.)
def load_model():
    """Load model once per process."""
    return {"weights": np.random.randn(100, 10)}

# Global model per process
_model = None

def init_worker():
    """Initialize model in worker process."""
    global _model
    _model = load_model()

def predict_batch(batch: np.ndarray) -> np.ndarray:
    """Run inference on a batch."""
    global _model
    if _model is None:
        _model = load_model()
    
    # Simulate inference
    time.sleep(0.1)
    return np.random.randn(len(batch), 10)

def parallel_inference(data: np.ndarray, batch_size: int = 32):
    """Run inference in parallel across processes."""
    # Split into batches
    batches = [
        data[i:i + batch_size]
        for i in range(0, len(data), batch_size)
    ]
    
    n_workers = mp.cpu_count()
    
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker
    ) as executor:
        results = list(executor.map(predict_batch, batches))
    
    return np.vstack(results)

if __name__ == '__main__':
    # Generate test data
    data = np.random.randn(1000, 100)
    
    start = time.perf_counter()
    predictions = parallel_inference(data, batch_size=64)
    elapsed = time.perf_counter() - start
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Time: {elapsed:.2f}s")
```

### GPU Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio
import queue
import threading
import time

class BatchInferenceService:
    """Accumulate requests into batches for efficient GPU inference."""
    
    def __init__(self, batch_size: int = 32, max_wait: float = 0.1):
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.request_queue = queue.Queue()
        self.running = True
        self._worker_thread = threading.Thread(target=self._worker)
        self._worker_thread.start()
    
    def _worker(self):
        """Process batches from queue."""
        while self.running:
            batch = []
            futures = []
            deadline = time.time() + self.max_wait
            
            # Collect batch
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    item, future = self.request_queue.get(timeout=0.01)
                    batch.append(item)
                    futures.append(future)
                except queue.Empty:
                    continue
            
            if batch:
                # Process batch
                results = self._predict_batch(batch)
                
                # Return results
                for future, result in zip(futures, results):
                    future.set_result(result)
    
    def _predict_batch(self, batch):
        """Run batch inference (simulate GPU work)."""
        time.sleep(0.05)  # Simulate GPU computation
        return [f"prediction_{i}" for i in range(len(batch))]
    
    def predict(self, input_data):
        """Submit single prediction, returns future."""
        future = asyncio.Future()
        self.request_queue.put((input_data, future))
        return future
    
    def shutdown(self):
        self.running = False
        self._worker_thread.join()
```

---

## Background Task Workers

### Async Task Queue

```python
import asyncio
from asyncio import Queue
from typing import Callable, Any

class AsyncTaskWorker:
    """Background worker for async tasks."""
    
    def __init__(self, n_workers: int = 5):
        self.n_workers = n_workers
        self.queue: Queue = Queue()
        self.results: dict[str, Any] = {}
        self._workers = []
    
    async def start(self):
        """Start worker tasks."""
        for i in range(self.n_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
    
    async def _worker(self, name: str):
        """Process tasks from queue."""
        while True:
            task_id, func, args, kwargs = await self.queue.get()
            try:
                result = await func(*args, **kwargs)
                self.results[task_id] = {"status": "success", "result": result}
            except Exception as e:
                self.results[task_id] = {"status": "error", "error": str(e)}
            finally:
                self.queue.task_done()
    
    async def submit(self, task_id: str, func: Callable, *args, **kwargs):
        """Submit task to queue."""
        await self.queue.put((task_id, func, args, kwargs))
    
    async def wait_all(self):
        """Wait for all tasks to complete."""
        await self.queue.join()
    
    def get_result(self, task_id: str):
        """Get result for task."""
        return self.results.get(task_id)

# Usage
async def process_item(item: str) -> str:
    await asyncio.sleep(0.5)
    return f"Processed: {item}"

async def main():
    worker = AsyncTaskWorker(n_workers=3)
    await worker.start()
    
    # Submit tasks
    for i in range(10):
        await worker.submit(f"task-{i}", process_item, f"item-{i}")
    
    # Wait for completion
    await worker.wait_all()
    
    # Get results
    for i in range(10):
        print(worker.get_result(f"task-{i}"))

asyncio.run(main())
```

---

## Real-Time Streaming Processor

### Event-Driven AI Pipeline

```python
import asyncio
from asyncio import Queue
from typing import AsyncIterator

class StreamProcessor:
    """Process streaming data with AI in real-time."""
    
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.input_queue: Queue = Queue()
        self.output_queue: Queue = Queue()
    
    async def process_item(self, item: dict) -> dict:
        """Process single item (e.g., AI inference)."""
        async with self.semaphore:
            await asyncio.sleep(0.1)  # Simulate processing
            return {
                "input": item,
                "output": f"processed_{item['id']}",
                "confidence": 0.95
            }
    
    async def producer(self, data: AsyncIterator[dict]):
        """Feed data into pipeline."""
        async for item in data:
            await self.input_queue.put(item)
        await self.input_queue.put(None)  # Signal done
    
    async def worker(self):
        """Process items from queue."""
        while True:
            item = await self.input_queue.get()
            if item is None:
                await self.input_queue.put(None)  # Pass signal
                break
            
            result = await self.process_item(item)
            await self.output_queue.put(result)
    
    async def consumer(self):
        """Collect and yield results."""
        while True:
            result = await self.output_queue.get()
            if result is None:
                break
            yield result
    
    async def run(self, data: AsyncIterator[dict], n_workers: int = 3):
        """Run the pipeline."""
        # Start workers
        workers = [
            asyncio.create_task(self.worker())
            for _ in range(n_workers)
        ]
        
        # Start producer
        producer = asyncio.create_task(self.producer(data))
        
        # Collect results
        results = []
        async for result in self.consumer():
            results.append(result)
            print(f"Got: {result['output']}")
        
        await producer
        await asyncio.gather(*workers)
        
        return results

# Usage
async def data_stream():
    for i in range(20):
        yield {"id": i, "data": f"item_{i}"}
        await asyncio.sleep(0.05)

async def main():
    processor = StreamProcessor(max_concurrent=5)
    results = await processor.run(data_stream(), n_workers=3)
    print(f"Processed {len(results)} items")

asyncio.run(main())
```

---

## Hands-on Exercise

### Your Task

```python
# Build an AI data pipeline that:
# 1. Reads data from files in parallel (I/O-bound)
# 2. Preprocesses text data in parallel (CPU-bound)
# 3. Sends batches to an LLM API concurrently (I/O-bound)
# 4. Collects and aggregates results
# 5. Reports progress throughout
```

<details>
<summary>✅ Solution</summary>

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import time

# Simulate file reading (I/O-bound)
def read_file(path: str) -> str:
    time.sleep(0.1)  # Simulate I/O
    return f"Content of {path}"

# Simulate preprocessing (CPU-bound)
def preprocess(text: str) -> dict:
    time.sleep(0.05)  # Simulate CPU work
    return {"text": text.lower(), "length": len(text)}

# Simulate LLM API call (I/O-bound)
async def call_llm(data: dict) -> dict:
    await asyncio.sleep(0.2)  # Simulate API latency
    return {"input": data, "response": "AI response"}

async def pipeline(file_paths: list[str], max_llm_concurrent: int = 5):
    """Complete AI data pipeline."""
    loop = asyncio.get_event_loop()
    
    # Phase 1: Read files (I/O-bound → threads)
    print("Phase 1: Reading files...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        read_tasks = [
            loop.run_in_executor(executor, read_file, path)
            for path in file_paths
        ]
        raw_data = await asyncio.gather(*read_tasks)
    print(f"  Read {len(raw_data)} files")
    
    # Phase 2: Preprocess (CPU-bound → processes)
    print("Phase 2: Preprocessing...")
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        preprocess_tasks = [
            loop.run_in_executor(executor, preprocess, text)
            for text in raw_data
        ]
        processed_data = await asyncio.gather(*preprocess_tasks)
    print(f"  Preprocessed {len(processed_data)} items")
    
    # Phase 3: LLM API calls (I/O-bound → async with semaphore)
    print("Phase 3: Calling LLM API...")
    semaphore = asyncio.Semaphore(max_llm_concurrent)
    
    async def limited_llm_call(data):
        async with semaphore:
            return await call_llm(data)
    
    llm_tasks = [limited_llm_call(data) for data in processed_data]
    
    # Track progress
    completed = 0
    results = []
    for coro in asyncio.as_completed(llm_tasks):
        result = await coro
        results.append(result)
        completed += 1
        if completed % 5 == 0:
            print(f"  Progress: {completed}/{len(llm_tasks)}")
    
    print(f"  Completed {len(results)} LLM calls")
    
    return results

if __name__ == '__main__':
    files = [f"file_{i}.txt" for i in range(20)]
    
    start = time.perf_counter()
    results = asyncio.run(pipeline(files, max_llm_concurrent=5))
    elapsed = time.perf_counter() - start
    
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Results: {len(results)} items processed")
```
</details>

---

## Summary

✅ **ProcessPoolExecutor** for CPU-bound preprocessing
✅ **ThreadPoolExecutor** for blocking I/O operations
✅ **asyncio + semaphore** for rate-limited API calls
✅ **Hybrid patterns** combine approaches for mixed workloads
✅ **Streaming processors** for real-time AI pipelines
✅ **Background workers** for async task queues

**Back to:** [Concurrency Overview](./00-concurrency-parallelism.md)

---

## Further Reading

- [LangChain Async](https://python.langchain.com/docs/concepts/async)
- [Ray for Distributed AI](https://docs.ray.io/)

<!-- 
Sources Consulted:
- Python concurrent.futures Docs: https://docs.python.org/3/library/concurrent.futures.html
-->
