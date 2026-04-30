# Week 65 — Local Deployment: Ollama, llama.cpp, and a CLI SQL Tool

## Learning Objectives

By the end of this week, you will be able to:

- Build llama.cpp from source on Apple Silicon with Metal acceleration enabled
- Register your Q4_K_M GGUF as an Ollama model with a custom system prompt
- Serve your model through Ollama's OpenAI-compatible REST API
- Write a CLI tool in Python that accepts a PostgreSQL schema and natural-language question and returns executable SQL
- Benchmark local latency end-to-end from prompt to first token

## Concepts

### llama.cpp on Apple Silicon

llama.cpp compiles to native Metal kernels on macOS, offloading matrix multiplications to the GPU on Apple Silicon. The critical build flag is `-DLLAMA_METAL=ON`. Without it, inference runs on CPU cores and is 3–5x slower.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_METAL=ON
cmake --build build -j$(sysctl -n hw.logicalcpu) --config Release
```

After building, `./build/bin/llama-cli` is your primary inference binary. Test with:

```bash
./build/bin/llama-cli \
    -m ~/models/postgres-sqlcoder-7b-Q4_K_M.gguf \
    -p "SELECT * FROM" \
    -n 50 \
    --n-gpu-layers 35
```

The `--n-gpu-layers` flag controls how many transformer layers run on Metal. For a 7B model with 32 layers, `--n-gpu-layers 32` (or 35 to include non-transformer tensors) puts everything on GPU. If you get Metal out-of-memory, reduce to 28.

### Ollama: Model Registry and Serving

Ollama is a local model server that wraps llama.cpp inference behind a clean REST API. It handles model loading, unloading, and conversation context management. You register your model with a Modelfile — a small config file analogous to a Dockerfile.

```dockerfile
# Modelfile for postgres-sqlcoder-7b
FROM /path/to/postgres-sqlcoder-7b-Q4_K_M.gguf

PARAMETER num_gpu 35
PARAMETER num_ctx 4096
PARAMETER temperature 0.1
PARAMETER stop "<|im_end|>"

SYSTEM """
You are a PostgreSQL expert. Given a database schema and a natural language question, write a correct and executable PostgreSQL query. Output only the SQL query with no explanation.
"""
```

Register and run:

```bash
ollama create postgres-sqlcoder -f Modelfile
ollama run postgres-sqlcoder "List all customers who placed orders in the last 7 days"
```

Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1`. This means any code that works with `openai.ChatCompletion` works with your local model by changing the `base_url`.

### Building a CLI SQL Tool

The CLI tool is a Python script that wraps the entire text-to-SQL pipeline in a usable terminal interface. It should accept a schema file and a natural-language question, format the prompt correctly (matching the training prompt format from Weeks 53–58), call Ollama, and print the resulting SQL.

```python
#!/usr/bin/env python3
"""
sql_ask.py — Ask your local postgres-sqlcoder model a SQL question.

Usage:
    python sql_ask.py --schema schema.sql "How many orders were placed yesterday?"
"""

import argparse
import sys
import httpx
import json

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "postgres-sqlcoder"

PROMPT_TEMPLATE = """### Task
Generate a SQL query to answer the following question.

### Database Schema
{schema}

### Question
{question}

### SQL Query
"""

def ask_sql(schema: str, question: str, temperature: float = 0.1) -> str:
    prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }
    response = httpx.post(OLLAMA_URL, json=payload, timeout=60.0)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def main():
    parser = argparse.ArgumentParser(description="NL→SQL via local postgres-sqlcoder")
    parser.add_argument("question", help="Natural language question")
    parser.add_argument("--schema", required=True, help="Path to SQL schema file")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    schema = open(args.schema).read()
    sql = ask_sql(schema, args.question, args.temperature)
    print(sql)

if __name__ == "__main__":
    main()
```

Key design decisions: the prompt template must exactly match the format your model was trained on (from your SFT dataset in Week 58). Using a different format will drop accuracy by 10–20 percentage points even with correct schema and question. The temperature of 0.1 is appropriate for deterministic SQL generation.

### Latency Profiling

For a local SQL assistant, latency is the user-facing metric. Measure two numbers:

- Time to first token (TTFT): how long the model takes to produce the first output token after receiving the prompt. This is the perceived lag for interactive use.
- Total generation time: time to complete the SQL query. For typical SQL (50–150 tokens), this is 1–4 seconds on M2 Pro at 35 GPU layers.

```python
import time

start = time.perf_counter()
# stream=True to capture first token time
with httpx.stream("POST", OLLAMA_URL, json={**payload, "stream": True}) as r:
    first_token_time = None
    for chunk in r.iter_lines():
        if chunk and first_token_time is None:
            first_token_time = time.perf_counter() - start
            print(f"TTFT: {first_token_time:.2f}s")
end = time.perf_counter()
print(f"Total: {end - start:.2f}s")
```

Target numbers on Apple Silicon M2 Pro: TTFT < 0.5s, total < 3s for 100-token SQL output.

## Connections

This week consumes the Q4_K_M GGUF produced in Week 64. The Ollama OpenAI-compatible API you set up here is the same interface pattern used in Week 66 for cloud deployment with vLLM. The CLI tool you build this week becomes a working demo artifact for your technical report (Week 67–70). The schema + question prompt format is your final deployment interface.

## Common Misconceptions / Pitfalls

The most common mistake is not matching the prompt format. Your model learned the specific start/end tokens and section structure from training data. The Modelfile SYSTEM prompt and the Python template must reproduce that format exactly — including the `### SQL Query` trigger line that tells the model to start outputting SQL.

Ollama's `num_ctx` must be set before creating the model, not after. If you create the model without setting `num_ctx 4096`, the default (2048) may truncate long schemas. Recreate the model with `ollama create` after updating the Modelfile.

Do not set `temperature=0.0` — most inference engines including Ollama treat this as "use default" rather than "greedy". Use `temperature=0.01` for near-greedy determinism.

Metal performance varies with macOS version. If you see unexpectedly slow Metal inference, update to macOS 14+ and rebuild llama.cpp.

## Time Allocation (6–8 hours)

- 1.0h: Build llama.cpp from source, verify Metal inference
- 1.0h: Install Ollama, write Modelfile, test with ollama run
- 1.5h: Write and test sql_ask.py CLI tool
- 1.0h: Latency profiling — TTFT and total generation time
- 1.0h: Accuracy spot-check on 20 custom examples through the CLI
- 0.5h: Clean up, write README for the tool, commit to your project repo
