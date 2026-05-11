# Week 65 Assignment Solutions

## Task 1: llama.cpp Metal Build

If Metal is not detected, check that you passed the cmake flag correctly and rebuild from scratch (delete the `build/` directory first).

```bash
# Clean rebuild with Metal
rm -rf build/
cmake -B build \
    -DLLAMA_METAL=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8

# Verify Metal in inference output
./build/bin/llama-cli \
    -m ~/models/postgres-sqlcoder-7b-Q4_K_M.gguf \
    -p "SELECT COUNT(*) FROM orders WHERE" \
    -n 80 \
    --n-gpu-layers 35 \
    2>&1 | head -30
# Look for: "llm_load_tensors: offloading 32 repeating layers to GPU"
# Look for: "llm_load_tensors: offloaded 35/35 layers to GPU"
```

Expected Metal tok/s on M2 Pro: 38–55. CPU-only: 8–14. Speedup: 4–5x.

## Task 2: Modelfile — Key Detail

The STOP token must match the one your model generates at end of turn. For Qwen2.5 chat models, this is `<|im_end|>`. Get this wrong and the model will keep generating beyond the SQL output.

```dockerfile
FROM /Users/raghuram/models/postgres-sqlcoder-7b-Q4_K_M.gguf

PARAMETER num_gpu 35
PARAMETER num_ctx 4096
PARAMETER temperature 0.1
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "###"

SYSTEM """You are a PostgreSQL SQL expert. When given a database schema and a question, output only a valid PostgreSQL SQL query. No explanation, no markdown code fences, no commentary. Just SQL."""
```

```bash
ollama create postgres-sqlcoder -f Modelfile
# Recreate if you change num_ctx — it is baked in at create time
ollama rm postgres-sqlcoder && ollama create postgres-sqlcoder -f Modelfile
```

## Task 3: CLI Tool — Complete sql_ask.py

```python
#!/usr/bin/env python3
import argparse, sys, httpx, json, time

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

TEMPLATE = """### Task
Generate a SQL query to answer the question below.

### Database Schema
{schema}

### Question
{question}

### SQL Query
"""

def ask(schema, question, model, temperature, stream=False):
    content = TEMPLATE.format(schema=schema, question=question)
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "temperature": temperature, "stream": stream}
    try:
        if stream:
            with httpx.stream("POST", OLLAMA_URL, json=payload, timeout=120) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        print(delta, end="", flush=True)
            print()
        else:
            r = httpx.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            print(r.json()["choices"][0]["message"]["content"].strip())
    except httpx.ConnectError:
        print("Error: Ollama is not running. Start it with: ollama serve", file=sys.stderr)
        sys.exit(1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--schema", required=True)
    p.add_argument("--model", default="postgres-sqlcoder")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--stream", action="store_true")
    args = p.parse_args()
    schema = open(args.schema).read()
    ask(schema, args.question, args.model, args.temperature, args.stream)

if __name__ == "__main__":
    main()
```

## Task 4: Latency Harness

```python
import httpx, json, time

def measure_ttft(prompt, model="postgres-sqlcoder"):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "stream": True, "temperature": 0.1}
    t0 = time.perf_counter()
    ttft = None
    with httpx.stream("POST", "http://localhost:11434/v1/chat/completions",
                      json=payload, timeout=120) as r:
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                if ttft is None:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return ttft, total
```

## Common Gotchas

- `ollama serve` must be running before any API call; it starts automatically on macOS if you installed via brew but not otherwise.
- Qwen2.5 uses a ChatML format; if your Modelfile SYSTEM prompt conflicts with the GGUF's embedded chat template, you may get doubled system prompts. Verify the output format with `llama-cli --verbose-prompt`.
- `num_ctx 4096` must be set in the Modelfile before `ollama create`. You cannot change it at inference time via the API.
- If `--n-gpu-layers 35` causes a Metal memory error, the GGUF may have more tensors than expected (e.g., tied embeddings count as extra). Try 32 first.

## How to Verify You Did It Right

Run this exact command and check output format:
```bash
echo "CREATE TABLE orders (id SERIAL PRIMARY KEY, amount DECIMAL, created_at TIMESTAMP);" > /tmp/schema.sql
python sql_ask.py --schema /tmp/schema.sql "What is the total revenue today?"
```
Expected output: a single SQL statement starting with `SELECT`, containing `SUM(amount)` or equivalent, and a `WHERE` clause filtering by today's date. No markdown, no explanation.
