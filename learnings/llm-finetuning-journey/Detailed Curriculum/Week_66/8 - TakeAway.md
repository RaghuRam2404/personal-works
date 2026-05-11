# Week 66 TakeAway — Cloud Deployment with vLLM

vLLM + paged attention + continuous batching turns one A10G into an SQL API that handles 120+ queries/minute at sub-second latency.

## Key Code Patterns

```bash
# Launch vLLM with AWQ (the flag is mandatory)
python -m vllm.entrypoints.openai.api_server \
    --model <handle>/postgres-sqlcoder-7b-awq-int4 \
    --quantization awq \
    --served-model-name postgres-sqlcoder \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

```python
# OpenAI client pointing at vLLM
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
resp = client.chat.completions.create(
    model="postgres-sqlcoder",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1, max_tokens=512)
```

```python
# Safe SQL output gate (always apply in production)
import sqlparse
def is_safe_sql(sql: str) -> bool:
    parsed = sqlparse.parse(sql.strip())
    if not parsed:
        return False
    return parsed[0].get_type() == "SELECT"
```

## Decision Rules

- If AWQ accuracy drops on specific SQL patterns: re-run AWQ with domain-enriched calibration set
- If cold-start is slow: cache model to persistent volume with `--download-dir`
- If GPU OOM at startup: reduce `--gpu-memory-utilization` to 0.85
- If concurrent throughput equals sequential: requests are not truly async — switch to `httpx.AsyncClient`
- Production SQL output: always parse and reject non-SELECT before execution

## Numbers to Remember

- A10G (24 GB): ~5 GB AWQ model + ~16 GB KV cache at 0.90 utilization
- Continuous batching gain: 3–5x aggregate tok/s at N=8 vs N=1
- RunPod A10G cost: ~$0.50–0.75/hr; $5 covers a full week of experiments
- vLLM cold-start (model already cached): ~30–60s; from Hub download: 3–5 min
- FastAPI + uvicorn overhead: < 5 ms per request (negligible vs model inference)

## Red Flags

- vLLM starts but generates garbage: forgot `--quantization awq` flag
- `# GPU blocks: 0` in vLLM startup log: model too large for remaining VRAM — reduce `gpu-memory-utilization` further
- Concurrent throughput ≈ sequential: asyncio calls are blocking — use async HTTP client
- API returns `DROP TABLE` SQL: no output validation layer — add sqlparse SELECT-only check immediately
- RunPod pod unreachable: port not exposed in pod's TCP settings before launch
