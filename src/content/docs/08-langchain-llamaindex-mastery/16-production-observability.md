---
title: "8.16 Production & Observability"
---

# 8.16 Production & Observability

## Introduction

Deploying LLM applications to production requires robust observability, error handling, and operational tooling. This section covers the production ecosystems of both LangChain (LangSmith) and LlamaIndex (LlamaCloud), along with framework-agnostic observability patterns essential for maintaining reliable AI systems.

## Learning Objectives

By the end of this section, you will be able to:
- Implement comprehensive tracing with LangSmith
- Use LlamaCloud services for document processing
- Set up OpenTelemetry for framework-agnostic observability
- Deploy agents with production-ready error handling
- Optimize costs and performance in production
- Implement security best practices

## LangSmith

### What is LangSmith?

LangSmith is LangChain's platform for:
- **Tracing**: Visualize every step of agent execution
- **Evaluation**: Test and benchmark LLM applications
- **Monitoring**: Track production performance
- **Debugging**: Understand failures and edge cases
- **Datasets**: Manage test cases and examples

### Setting Up LangSmith

**Environment Variables**:
- Set `LANGCHAIN_TRACING_V2` to "true" to enable tracing
- Set `LANGCHAIN_API_KEY` to your LangSmith API key
- Set `LANGCHAIN_PROJECT` to your project name (e.g., "my-production-app")

**Automatic Tracing**:
- Once environment variables are set, all LangChain/LangGraph code is traced automatically
- Import and use `ChatOpenAI` from `langchain_openai`
- Import `StateGraph` from `langgraph.graph`
- All invocations send traces to LangSmith without additional code

### Tracing LangGraph Agents

**Dependencies**:
- `StateGraph`, `START`, `END` from `langgraph.graph`
- `traceable` decorator from `langsmith`

**State Definition (AgentState)**:
- `messages`: List for conversation history
- `result`: String for final output

**Custom Traced Functions**:
- Use `@traceable(run_type="chain")` decorator on custom functions
- The `run_type` specifies how the function appears in traces
- These functions appear as separate spans in LangSmith UI

**Agent Node**:
- Call traced functions within node: node execution is also traced
- All nested calls appear in the trace hierarchy

**Graph Setup**:
- Create `StateGraph(AgentState)`, add nodes and edges as normal
- Compile with `workflow.compile()`

**Explicit Tracing Metadata**:
- Import `trace` from `langsmith`
- Use `with trace("run-name", project_name="my-app") as t:` context
- Set metadata: `t.metadata = {"user_id": "123", "session": "abc"}`
- Invoke app inside the context for tagged traces

### Evaluation with LangSmith

**Dependencies**:
- `Client` from `langsmith`
- `evaluate` from `langsmith.evaluation`

**Creating a Dataset**:
- Initialize client: `client = Client()`
- Create dataset: `dataset = client.create_dataset("qa-evaluation")`

**Adding Examples**:
- Call `client.create_examples(...)` with:
  - `inputs`: List of input dictionaries (e.g., `[{"question": "What is AI?"}]`)
  - `outputs`: List of expected output dictionaries (e.g., `[{"answer": "..."}]`)
  - `dataset_id`: The dataset's ID from creation

**Defining Evaluators**:
- Create function accepting `run` and `example` parameters
- Extract predicted output: `run.outputs["answer"]`
- Extract expected output: `example.outputs["answer"]`
- Use LLM-as-judge or other scoring method
- Return dict with `score` (float) and `key` (string identifier)

**Running Evaluation**:
- Call `evaluate(...)` with:
  - First arg: Lambda wrapping app invocation
  - `data`: Dataset name string
  - `evaluators`: List of evaluator functions
  - `experiment_prefix`: Version identifier (e.g., "v1.0")
- Results stored in LangSmith for comparison

### Production Monitoring

**Dependencies**:
- `Client` from `langsmith`

**Querying Runs**:
- Initialize client: `client = Client()`
- Call `client.list_runs(...)` with filters:
  - `project_name`: Your project name
  - `filter`: Query string with conditions (e.g., `'and(eq(status, "error"), gte(start_time, "2024-01-01"))'`)
  - `limit`: Maximum number of runs to return

**Analyzing Results**:
- Iterate over runs to access:
  - `run.error`: Error message if failed
  - `run.inputs`: The input data that caused the issue
  - `run.trace_id`: Unique identifier for the full trace

**Use Cases**:
- Debug production errors
- Identify problematic input patterns
- Track error rates over time

## LlamaCloud Ecosystem

### LlamaParse for Document Processing

Complex document parsing with structure preservation:

**Dependencies**:
- `LlamaParse` from `llama_cloud`

**Parser Initialization**:
- Create parser with configuration:
  - `api_key`: Your LlamaCloud API key
  - `result_type`: Output format ("markdown", "text", or "json")
  - `parsing_instruction`: Custom instructions (e.g., "Extract all tables and figures with captions")

**Single Document Parsing**:
- Call `parser.load_data("filename.pdf")` with file path
- Returns list of parsed documents

**Batch Processing**:
- Pass list of file paths: `parser.load_data(["report1.pdf", "report2.pdf", "presentation.pptx"])`
- Processes multiple files in one call

**Accessing Parsed Content**:
- Iterate over documents: `for doc in documents:`
- Access text: `doc.text`
- Access metadata: `doc.metadata` (contains tables, images, structure info)

### LlamaExtract for Structured Extraction

Schema-based extraction from documents:

**Dependencies**:
- `LlamaExtract` from `llama_cloud`
- `BaseModel` from `pydantic`

**Defining Extraction Schema**:
- Create Pydantic model class (e.g., `InvoiceData(BaseModel)`)
- Define fields with types:
  - `invoice_number`: str
  - `date`: str
  - `total_amount`: float
  - `line_items`: list[dict]
  - `vendor_name`: str

**Extraction Process**:
- Create extractor: `extractor = LlamaExtract(api_key="your-key")`
- Call `extractor.extract(documents=["invoice.pdf"], schema=InvoiceData)`
- Returns list of typed objects matching your schema

**Accessing Extracted Data**:
- Get first result: `invoice = result[0]`
- Access typed fields: `invoice.invoice_number`, `invoice.total_amount`
- Full Pydantic validation and typing support

### LlamaCloud Managed Index

**Dependencies**:
- `LlamaCloudIndex` from `llama_cloud`

**Creating Managed Index**:
- Call `LlamaCloudIndex.create(...)` with:
  - `name`: Index identifier (e.g., "production-knowledge-base")
  - `api_key`: Your LlamaCloud API key
  - `embedding_model`: Model name (e.g., "text-embedding-3-small")
  - `chunk_size`: Document chunk size (e.g., 512)

**Adding Documents**:
- Call `index.add_documents(documents)` with document list

**Querying**:
- Call `index.query("Your question here")`
- Serverless execution - scales automatically
- Returns relevant results without infrastructure management

## OpenTelemetry Integration

### Framework-Agnostic Observability

**Dependencies**:
- `trace` from `opentelemetry`
- `TracerProvider` from `opentelemetry.sdk.trace`
- `BatchSpanProcessor` from `opentelemetry.sdk.trace.export`
- `OTLPSpanExporter` from `opentelemetry.exporter.otlp.proto.grpc.trace_exporter`

**Tracer Setup**:
- Create provider: `provider = TracerProvider()`
- Create exporter with collector endpoint: `OTLPSpanExporter(endpoint="http://otel-collector:4317")`
- Create processor: `processor = BatchSpanProcessor(exporter)`
- Add processor to provider: `provider.add_span_processor(processor)`
- Set global provider: `trace.set_tracer_provider(provider)`
- Get tracer: `tracer = trace.get_tracer(__name__)`

### Instrumenting LLM Calls

**Dependencies**:
- `trace` from `opentelemetry`

**Creating Traced Function**:
- Get tracer: `tracer = trace.get_tracer(__name__)`
- Define async function for LLM calls

**Span Creation**:
- Use `with tracer.start_as_current_span("llm_completion") as span:`
- All code inside the context is traced

**Setting Span Attributes**:
- Before call: `span.set_attribute("prompt.length", len(prompt))`
- Before call: `span.set_attribute("model", "gpt-4o")`

**Capturing Metrics**:
- Track timing: `start = time.time()`, `duration = time.time() - start`
- After call: `span.set_attribute("response.length", len(response.text))`
- After call: `span.set_attribute("duration_ms", duration * 1000)`
- After call: `span.set_attribute("tokens.total", response.usage.total_tokens)`

### LlamaIndex Instrumentation

**Dependencies**:
- `Settings` from `llama_index.core`
- `CallbackManager` from `llama_index.core.callbacks`
- `get_dispatcher` from `llama_index.core.instrumentation`

**Built-in Instrumentation**:
- Get dispatcher: `dispatcher = get_dispatcher()`

**Custom Span Handler Class**:
- Create class with tracer instance in constructor
- Implement `span_enter(self, id_, bound_args, instance, parent_id)` method:
  - Start new span with tracer
  - Return span object
- Implement `span_exit(self, id_, bound_args, instance, result, span)` method:
  - End the span

**Registering Handler**:
- Call `dispatcher.add_span_handler(OTelSpanHandler(tracer))`
- All LlamaIndex operations now emit OpenTelemetry spans

## Production Deployment

### LangGraph Cloud

**Configuration File (langgraph.json)**:
- `dependencies`: List of pip packages (e.g., `["langchain-openai", "langgraph"]`)
- `graphs`: Object mapping graph names to module paths (e.g., `{"agent": "./agent.py:graph"}`)
- `env`: Path to environment file (e.g., `".env"`)

**CLI Deployment**:
- Run `langgraph cloud deploy` command to deploy

**SDK Usage**:
- Import `get_client` from `langgraph_sdk`
- Create client: `client = get_client(url="https://your-deployment.langchain.app")`

**Running Deployed Agent**:
- Use `async for event in client.runs.stream(...):`
- Parameters:
  - `assistant_id`: Graph name (e.g., "agent")
  - `input`: Input state dict (e.g., `{"messages": [{"role": "user", "content": "Hello"}]}`)
- Events stream in real-time as agent executes

### llama_deploy for LlamaIndex

**Dependencies**:
- `deploy_workflow`, `ControlPlaneConfig`, `WorkflowServiceConfig`, `SimpleMessageQueueConfig` from `llama_deploy`

**Deploy Control Plane**:
- Call `await deploy_core(...)` with:
  - `control_plane_config`: `ControlPlaneConfig(host="0.0.0.0", port=8000)`
  - `message_queue_config`: `SimpleMessageQueueConfig(host="rabbitmq", port=5672)`

**Deploy Workflow Service**:
- Call `await deploy_workflow(...)` with:
  - `workflow`: Your workflow instance (e.g., `ProductionRAGWorkflow()`)
  - `config`: `WorkflowServiceConfig` with:
    - `host`: Server address ("0.0.0.0")
    - `port`: Service port (8001)
    - `service_name`: Identifier ("rag-service")
    - `num_workers`: Worker count for scaling (4)

### Docker Deployment

**Dockerfile Structure**:
- Base image: `python:3.11-slim`
- Set working directory: `WORKDIR /app`
- Copy and install requirements: `COPY requirements.txt .` then `RUN pip install -r requirements.txt`
- Copy application code: `COPY . .`
- Set environment variables:
  - `ENV LANGCHAIN_TRACING_V2=true`
  - `ENV LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}` (from build args)
- Entry command: `CMD ["python", "-m", "uvicorn", "app:api", "--host", "0.0.0.0", "--port", "8000"]`

**Docker Compose Configuration**:
- Version: '3.8'
- Services:
  - `agent` service:
    - `build: .` - Build from Dockerfile
    - `ports: ["8000:8000"]` - Expose port
    - `environment`: Pass API keys from host environment
    - `deploy.replicas: 3` - Run 3 instances for scaling
    - `healthcheck`: Configure with test command, interval, timeout, retries
  - `redis` service:
    - `image: redis:alpine`
    - `ports: ["6379:6379"]`
    - Used for caching and session storage

## Error Handling Patterns

### Retry Strategies

**Dependencies**:
- `retry`, `stop_after_attempt`, `wait_exponential`, `retry_if_exception_type` from `tenacity`
- `openai` module for exception types

**Retry Decorator Configuration**:
- `@retry(...)` decorator with:
  - `stop=stop_after_attempt(3)`: Maximum 3 attempts
  - `wait=wait_exponential(multiplier=1, min=4, max=60)`: Exponential backoff starting at 4s, max 60s
  - `retry=retry_if_exception_type(...)`: Only retry specific exceptions

**Retryable Exceptions**:
- `openai.RateLimitError`: API rate limit exceeded
- `openai.APIConnectionError`: Network connectivity issues
- Pass as tuple to `retry_if_exception_type()`

**Function Structure**:
- Async function that makes the LLM call
- Automatic retry on specified transient errors

### Fallback Patterns

**FallbackLLM Class Structure**:

*Constructor*:
- Accept `primary_model` and `fallback_model` strings
- Create two `ChatOpenAI` instances: `self.primary` and `self.fallback`

*invoke Method (async)*:
- Try block: Call `await self.primary.ainvoke(messages)`
- Except block: Log warning, call `await self.fallback.ainvoke(messages)`
- Returns response from whichever model succeeds

**Usage Example**:
- Create instance: `llm = FallbackLLM(primary_model="gpt-4o", fallback_model="gpt-3.5-turbo")`
- Automatically falls back to cheaper/faster model on primary failure

### Graceful Degradation

**Multi-Level Fallback Pattern**:
- Try full RAG pipeline first: `return await full_rag_pipeline(query)`
- Catch `VectorStoreError`: Log error, fall back to keyword search
- Catch `LLMError`: Log error, return cached/template response

**Fallback Hierarchy**:
1. Primary: Full RAG with vector search and LLM
2. Secondary: Keyword-based search pipeline (no vector DB needed)
3. Tertiary: Pre-cached responses or template answers

**Benefits**:
- Always returns some response to user
- Maintains service availability during partial outages
- Degrades quality gracefully rather than failing completely

## Cost Optimization

### Token Usage Tracking

**Dependencies**:
- `get_openai_callback` from `langchain.callbacks`

**Callback Context Manager**:
- Use `with get_openai_callback() as cb:` context
- All LangChain LLM calls inside are tracked

**Accessing Metrics**:
- `cb.total_tokens`: Total tokens used
- `cb.prompt_tokens`: Input/prompt tokens
- `cb.completion_tokens`: Output/completion tokens
- `cb.total_cost`: Estimated cost in USD

**Recording to Monitoring System**:
- Call `metrics.record("llm_tokens", cb.total_tokens)`
- Call `metrics.record("llm_cost", cb.total_cost)`
- Enables dashboards and alerting on usage

### Caching Strategies

**Dependencies**:
- `RedisCache` from `langchain.cache`
- `set_llm_cache` from `langchain.globals`
- `redis` module

**Redis Cache Setup**:
- Create Redis client: `redis_client = redis.Redis(host="localhost", port=6379)`
- Create cache: `RedisCache(redis_client)`
- Set globally: `set_llm_cache(cache)`

**Cache Behavior**:
- First call with a prompt: Hits the API, caches response
- Subsequent identical prompts: Returns cached response
- Exact string match required for cache hit

### Semantic Caching

**Dependencies**:
- `RedisSemanticCache` from `langchain.cache`
- `OpenAIEmbeddings` from `langchain_openai`

**Semantic Cache Setup**:
- Create cache with `RedisSemanticCache(...)` passing:
  - `redis_url`: Connection string (e.g., "redis://localhost:6379")
  - `embedding`: Embedding model instance (e.g., `OpenAIEmbeddings()`)
  - `score_threshold`: Similarity threshold (e.g., 0.95 for 95% similarity)
- Set globally with `set_llm_cache(cache)`

**Semantic Cache Behavior**:
- Embeds prompts and compares similarity
- Semantically similar queries can hit cache
- Example: "What is artificial intelligence?" and "Define AI" may share cache
- Higher threshold = stricter matching

## Security Best Practices

### API Key Management

**Dependencies**:
- `BaseSettings` from `pydantic_settings`

**Settings Class**:
- Create class extending `BaseSettings`
- Define fields as typed strings:
  - `openai_api_key: str`
  - `langchain_api_key: str`
  - `database_url: str`
- Configure inner `Config` class:
  - `env_file = ".env"`: Load from .env file
  - `env_file_encoding = "utf-8"`: File encoding

**Usage**:
- Create instance: `settings = Settings()`
- Access values: `settings.openai_api_key`
- Values automatically loaded from environment or .env file

**Security**:
- Silence HTTP logging: `logging.getLogger("httpx").setLevel(logging.WARNING)`
- Prevents accidental API key exposure in logs

### Input Validation

**Dependencies**:
- `BaseModel`, `validator` from `pydantic`
- `re` module for regex

**UserQuery Model**:
- `query: str` - The user's query text
- `user_id: str` - User identifier

**Query Validator**:
- Use `@validator("query")` decorator
- Remove dangerous characters: `re.sub(r'[<>{}]', '', v)`
- Enforce length limit: Raise `ValueError` if len > 1000
- Return sanitized value

**User ID Validator**:
- Use `@validator("user_id")` decorator
- Validate format with regex: `^[a-zA-Z0-9_-]+$`
- Raise `ValueError` for invalid format
- Prevents injection via user_id field

**Benefits**:
- Automatic validation on model creation
- Prevents prompt injection attacks
- Enforces input constraints

### Output Sanitization

**Function: sanitize_llm_output(output: str) -> str**

**Code Execution Prevention**:
- Use regex to find Python code blocks with `import os`
- Pattern: `r'```python\s*import os.*?```'` with `re.DOTALL` flag
- Replace with `[code removed]` placeholder

**PII Redaction**:
- SSN pattern: `r'\b\d{3}-\d{2}-\d{4}\b`
- Replace with `[SSN REDACTED]`
- Credit card pattern: `r'\b\d{16}\b`
- Replace with `[CARD REDACTED]`

**Return Value**:
- Sanitized string safe for display to users

**Use Cases**:
- Prevent code execution instructions in output
- Comply with data privacy regulations
- Protect sensitive data from accidental exposure

### Audit Logging

**Dependencies**:
- `structlog` module
- `datetime` from datetime

**Logger Setup**:
- Get structured logger: `logger = structlog.get_logger()`

**Audited Query Function**:

*Entry Logging*:
- Generate unique `trace_id`
- Log "query_received" event with:
  - `trace_id`: For correlation
  - `user_id`: Who made the request
  - `query_hash`: Hash of query (not raw content for privacy)
  - `timestamp`: ISO format datetime

*Success Logging*:
- Log "query_completed" event with:
  - `trace_id`: Same as entry
  - `success`: True
  - `response_length`: Size of response

*Error Logging*:
- Log "query_failed" event with:
  - `trace_id`: Same as entry
  - `error`: Error message string
  - `error_type`: Exception class name
- Re-raise the exception after logging

**Benefits**:
- Full audit trail for compliance
- Correlation via trace_id
- Structured format for log analysis

## Hands-On Exercise

### Build a Production-Ready Agent

Implement a complete production setup:

1. LangGraph agent with checkpointing
2. LangSmith tracing
3. Retry logic with fallbacks
4. Token tracking and cost limits
5. Input validation
6. Audit logging

**Exercise Requirements**:

*Dependencies*:
- `StateGraph` from `langgraph.graph`
- `traceable` from `langsmith`
- `retry` from `tenacity`
- `structlog` for logging

*ProductionAgent Class*:

*Constructor*:
- Accept `max_cost_per_query: float = 0.10` parameter
- Store as `self.max_cost`
- Initialize structured logger: `self.logger = structlog.get_logger()`
- Set up checkpointer, LLM clients, and validators

*query Method*:
- Decorated with `@traceable` for LangSmith
- Decorated with `@retry(...)` for resilience
- Accept `user_query: UserQuery` (Pydantic validated input)
- Return `AgentResponse` typed result
- Implement token tracking with cost limit enforcement
- Include audit logging for all operations

## Key Takeaways

1. **LangSmith** provides comprehensive tracing and evaluation
2. **LlamaCloud** offers production document processing
3. **OpenTelemetry** enables vendor-neutral observability
4. **Retry patterns** handle transient failures
5. **Caching** dramatically reduces costs
6. **Security** requires input validation and output sanitization
7. **Audit logging** is essential for compliance

## Additional Resources

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LlamaCloud Documentation](https://cloud.llamaindex.ai/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Production LLM Apps Guide](https://www.langchain.com/production)

---

*This concludes Unit 8: LangChain & LlamaIndex Mastery*

*Estimated section study time: 5-7 hours*
*Hands-on practice: 4+ hours*
