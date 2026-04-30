# Week 66 Glossary — Cloud Deployment with vLLM

**vLLM**: An open-source LLM inference server implementing paged attention and continuous batching for high-throughput GPU serving.

**Paged attention**: KV cache management scheme that allocates memory in fixed-size pages on demand, eliminating fragmentation and enabling higher concurrency than pre-allocated contiguous blocks.

**Continuous batching**: Inference scheduling strategy that adds new requests to an ongoing batch mid-generation, rather than waiting for all sequences to complete before starting new ones.

**`--gpu-memory-utilization`**: vLLM flag controlling what fraction of total GPU VRAM is reserved for model weights plus KV cache (recommended: 0.90).

**`--served-model-name`**: vLLM flag that sets the model name string returned in API responses; allows the endpoint to advertise a custom name instead of the HuggingFace repo path.

**Paged KV cache**: The physical implementation of paged attention: KV entries are stored in a pool of fixed-size blocks mapped by a per-sequence page table.

**Cold-start latency**: Time from pod power-on to first successful inference request; dominated by model weight download (from Hub) and GPU loading.

**FastAPI**: A modern Python async web framework used to build the SQL API wrapper; backed by Pydantic for request/response schema validation.

**Uvicorn**: ASGI server that runs FastAPI applications; use `uvicorn app:app --host 0.0.0.0 --port 9000` to expose the API.

**Server-Sent Events (SSE)**: HTTP streaming protocol used by both Ollama and vLLM for token-by-token streaming responses; each chunk is a `data: {...}` line.

**sqlparse**: Python library for parsing SQL strings into an AST; used to detect and reject non-SELECT statements in the API safety layer.
