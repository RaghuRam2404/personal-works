# Week 66 — Cloud Deployment: vLLM, OpenAI-Compatible API, HF Spaces

## Learning Objectives

By the end of this week, you will be able to:

- Launch a vLLM server on a RunPod GPU instance serving your AWQ INT4 model
- Configure vLLM for continuous batching and paged attention for high-throughput inference
- Test the deployment through the OpenAI-compatible `/v1/chat/completions` endpoint
- Build a minimal FastAPI wrapper that adds schema injection and SQL post-processing
- Optionally deploy a Gradio demo to Hugging Face Spaces

## Concepts

### vLLM and Continuous Batching

vLLM is the production-grade inference server for LLMs on GPU. Its two core innovations are paged attention (managing KV cache as virtual memory pages to eliminate fragmentation) and continuous batching (dynamically adding new requests to an ongoing batch mid-inference, rather than waiting for all sequences to finish).

For your SQL model, the practical implication is that vLLM handles multiple concurrent requests far more efficiently than a naive Python loop. A single A10G (24 GB) running your AWQ INT4 model (~5 GB weights) has ~19 GB left for KV cache — enough for hundreds of concurrent 4096-token contexts in paged form.

Install and launch are straightforward:

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model <your-handle>/postgres-sqlcoder-7b-awq-int4 \
    --quantization awq \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

The `--gpu-memory-utilization 0.90` tells vLLM to allocate 90% of GPU VRAM for weights + KV cache combined. The remaining 10% is reserved for CUDA overhead and temporary buffers.

### Running on RunPod

RunPod provides GPU instances with an SSH tunnel and optional HTTP endpoint exposure. The workflow for cloud deployment:

1. Launch a RunPod instance with an A10G (24 GB) or RTX 3090 (24 GB). Select the "RunPod PyTorch" template.
2. In the pod's network settings, expose TCP port 8000 and note the public IP.
3. SSH into the pod, install vLLM, and launch the server.
4. From your local machine, point your API client at `http://<pod-ip>:<port>/v1/`.

For a shared demo, use RunPod's "serverless" option or expose via a Cloudflare Tunnel to get a stable HTTPS URL.

```bash
# On RunPod pod
pip install vllm huggingface_hub
huggingface-cli login --token $HF_TOKEN
python -m vllm.entrypoints.openai.api_server \
    --model <your-handle>/postgres-sqlcoder-7b-awq-int4 \
    --quantization awq \
    --served-model-name postgres-sqlcoder \
    --max-model-len 4096 \
    --port 8000 &

# On local machine — test with openai client
from openai import OpenAI
client = OpenAI(base_url="http://<pod-ip>:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="postgres-sqlcoder",
    messages=[{"role": "user", "content": your_prompt}],
    temperature=0.1,
)
print(resp.choices[0].message.content)
```

### Building a FastAPI Wrapper

Exposing vLLM directly works for development, but production deployments benefit from an API layer that: injects the schema automatically, validates the SQL output, and provides structured JSON responses.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import sqlparse, re

app = FastAPI()
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

SCHEMA = open("schema.sql").read()  # loaded once at startup

TEMPLATE = """### Task
{question}

### Database Schema
{schema}

### SQL Query
"""

class SQLRequest(BaseModel):
    question: str
    temperature: float = 0.1

class SQLResponse(BaseModel):
    sql: str
    tokens_used: int

def clean_sql(raw: str) -> str:
    # Strip markdown fences
    raw = re.sub(r"```(?:sql)?\n?", "", raw).strip().rstrip("`").strip()
    return raw

@app.post("/sql", response_model=SQLResponse)
async def generate_sql(req: SQLRequest):
    prompt = TEMPLATE.format(question=req.question, schema=SCHEMA)
    resp = client.chat.completions.create(
        model="postgres-sqlcoder",
        messages=[{"role": "user", "content": prompt}],
        temperature=req.temperature,
        max_tokens=512,
    )
    raw_sql = resp.choices[0].message.content
    sql = clean_sql(raw_sql)
    tokens = resp.usage.total_tokens
    return SQLResponse(sql=sql, tokens_used=tokens)
```

Run with `uvicorn app:app --host 0.0.0.0 --port 9000`. Now callers POST `{"question": "How many orders yesterday?"}` and receive clean SQL in a typed response.

### HuggingFace Spaces (Optional)

HuggingFace Spaces with a Gradio SDK lets you deploy a public demo with zero server management. The constraint is that free Spaces use CPU-only, which is too slow for BF16 but acceptable for Q4_K_M GGUF via llama-cpp-python.

```python
# app.py for HF Space
import gradio as gr
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="<your-handle>/postgres-sqlcoder-7b-Q4_K_M-GGUF",
    filename="postgres-sqlcoder-7b-Q4_K_M.gguf",
    n_ctx=4096, n_threads=4,
)

def generate(schema, question):
    prompt = f"### Task\n{question}\n### Database Schema\n{schema}\n### SQL Query\n"
    out = llm(prompt, max_tokens=256, temperature=0.1, stop=["###"])
    return out["choices"][0]["text"].strip()

gr.Interface(fn=generate,
             inputs=[gr.Textbox(label="Schema"), gr.Textbox(label="Question")],
             outputs=gr.Code(language="sql")).launch()
```

## Connections

The cloud deployment pattern you establish this week is the endpoint that Week 76 (agentic SQL) will call when building multi-step query pipelines. The FastAPI wrapper structure is referenced in the technical report (Week 67–70) as your deployment architecture section. The Spaces demo is a useful artifact to link from your report.

## Common Misconceptions / Pitfalls

vLLM does not support all quantization formats equally. Ensure you pass `--quantization awq` when loading your AWQ model — omitting this flag causes vLLM to try loading the model as BF16, which may OOM or produce garbled output because the weight shapes have been remapped.

`gpu-memory-utilization` above 0.95 often causes CUDA out-of-memory during the KV cache preallocation step even if the weights fit. Stay at 0.90.

On RunPod, the pod IP changes every time you restart the pod. Use RunPod's "Serverless" or set up a Cloudflare Tunnel for a stable URL if you share the demo.

The OpenAI Python client requires `api_key` to be non-empty even for local endpoints. Pass `api_key="not-needed"` or any non-empty string — vLLM ignores it by default.

## Time Allocation (6–8 hours)

- 0.5h: Provision RunPod A10G instance
- 1.0h: Install vLLM, download model, launch server, verify endpoint
- 1.0h: Test throughput — concurrent requests, measure tok/s vs single-request baseline
- 1.5h: Build FastAPI wrapper with schema injection and SQL cleaning
- 1.0h: Write integration test: 20 questions through the API, check accuracy
- 1.0h: Optional — HF Spaces Gradio demo with GGUF via llama-cpp-python
- 0.5h: Document API, write README, add costs to `week66_results.md`
