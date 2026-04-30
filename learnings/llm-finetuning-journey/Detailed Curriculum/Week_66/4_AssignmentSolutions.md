# Week 66 Assignment Solutions

## Task 1: vLLM Launch — Key Details

The most common failure is forgetting `--quantization awq`. Without it, vLLM tries to load the model as BF16 and either OOMs or produces garbage because the quantized weight tensors have a different layout than what BF16 loading expects.

```bash
# Full launch command with all recommended flags
python -m vllm.entrypoints.openai.api_server \
    --model <your-handle>/postgres-sqlcoder-7b-awq-int4 \
    --quantization awq \
    --dtype float16 \
    --served-model-name postgres-sqlcoder \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --disable-log-requests \
    --port 8000

# Quick smoke test
python -c "
from openai import OpenAI
c = OpenAI(base_url='http://localhost:8000/v1', api_key='x')
r = c.chat.completions.create(model='postgres-sqlcoder',
    messages=[{'role':'user','content':'SELECT count of users'}], temperature=0.1)
print(r.choices[0].message.content)
"
```

## Task 2: Concurrent Load Test

```python
import asyncio, httpx, time

QUESTIONS = [f"Count orders for customer {i}" for i in range(16)]
SCHEMA = open("schema.sql").read()
URL = "http://localhost:8000/v1/chat/completions"
PROMPT_TMPL = "### Task\n{q}\n### Database Schema\n{s}\n### SQL Query\n"

async def single_request(client, question):
    payload = {
        "model": "postgres-sqlcoder",
        "messages": [{"role": "user",
                      "content": PROMPT_TMPL.format(q=question, s=SCHEMA)}],
        "temperature": 0.1, "max_tokens": 256,
    }
    t0 = time.perf_counter()
    r = await client.post(URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    latency = time.perf_counter() - t0
    tokens = data["usage"]["completion_tokens"]
    return latency, tokens

async def load_test(n):
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[single_request(client, QUESTIONS[i]) for i in range(n)])
        total = time.perf_counter() - t0
    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)
    print(f"N={n}: total={total:.1f}s avg_lat={sum(latencies)/n:.1f}s tok/s={total_tokens/total:.0f}")

for n in [1, 4, 8, 16]:
    asyncio.run(load_test(n))
```

Expected: at N=16, aggregate tok/s should be 3–5x higher than N=1 due to continuous batching.

## Task 3: FastAPI app.py — Complete Implementation

```python
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import re, time, logging

app = FastAPI()
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
SCHEMA = open("schema.sql").read()
TMPL = "### Task\n{q}\n### Database Schema\n{s}\n### SQL Query\n"

class Req(BaseModel):
    question: str
    temperature: float = 0.1

class Resp(BaseModel):
    sql: str
    tokens_used: int

def clean(raw: str) -> str:
    raw = re.sub(r"```(?:sql)?\n?", "", raw)
    raw = raw.replace("```", "").strip()
    return raw

@app.get("/health")
def health():
    return {"status": "ok", "model": "postgres-sqlcoder"}

@app.post("/sql", response_model=Resp)
def gen_sql(req: Req):
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model="postgres-sqlcoder",
        messages=[{"role": "user", "content": TMPL.format(q=req.question, s=SCHEMA)}],
        temperature=req.temperature, max_tokens=512,
    )
    sql = clean(r.choices[0].message.content)
    tokens = r.usage.total_tokens
    logging.info(f"Q: {req.question[:50]} | tok: {tokens} | t: {time.perf_counter()-t0:.2f}s")
    return Resp(sql=sql, tokens_used=tokens)
```

## Common Gotchas

- RunPod exposes ports only if you add them to the "TCP Port Exposures" list before starting the pod; you cannot add them after start.
- `gpu-memory-utilization=0.95` often causes OOM at KV cache preallocation even if weights fit; stay at 0.90.
- The OpenAI client requires `base_url` to end without a trailing slash: `http://host:8000/v1` not `http://host:8000/v1/`.
- vLLM logs the GPU memory allocation plan at startup — check that it allocates KV cache blocks (you should see `# GPU blocks: N`); if N is 0, the model is too large for the remaining VRAM.
- On RunPod, the pod public IP is under "Connect" → "TCP Port" → copy the hostname:port pair, not just the IP.

## How to Verify You Did It Right

```bash
curl -X POST http://<your-pod-ip>:9000/sql \
     -H "Content-Type: application/json" \
     -d '{"question": "How many orders were placed in the last 24 hours?"}'
# Expected response:
# {"sql":"SELECT COUNT(*) FROM orders WHERE created_at >= NOW() - INTERVAL '24 hours'","tokens_used":87}
```

The SQL field should contain clean, valid PostgreSQL with no markdown and no trailing explanation.
