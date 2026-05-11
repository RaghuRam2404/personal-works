# Week 65 Assignment — Local Deployment: Ollama + CLI Tool

## Setup Checklist

- [ ] `postgres-sqlcoder-7b-Q4_K_M.gguf` available locally (from Week 64)
- [ ] llama.cpp cloned and built with Metal: `cmake -B build -DLLAMA_METAL=ON && cmake --build build -j8`
- [ ] Ollama installed: `brew install ollama` or download from `https://ollama.com`
- [ ] Python environment with `httpx` installed: `pip install httpx`
- [ ] A PostgreSQL schema file for testing (your TimescaleDB orders schema from earlier weeks, or create a simple one)

## Task 1: Build and Smoke-Test llama.cpp with Metal

**Goal:** Confirm your GGUF runs on Metal GPU layers with acceptable speed.

**Requirements:**
- [ ] Build llama.cpp with `-DLLAMA_METAL=ON`; confirm `ggml_metal_init` appears in startup logs
- [ ] Run inference with `--n-gpu-layers 35` and confirm Metal is used (log line: `llm_load_tensors: offloading N layers to GPU`)
- [ ] Measure baseline tok/s: run `llama-cli -m model.gguf -p "SELECT" -n 100 --n-gpu-layers 35` and record reported `eval time` tokens/sec
- [ ] Compare to CPU-only: run same command without `--n-gpu-layers` and record speedup ratio
- [ ] Target: Metal should be at least 3x faster than CPU-only on M1/M2

**Deliverable:** `week65_results.md` with Metal tok/s, CPU tok/s, and speedup ratio

## Task 2: Register Model in Ollama

**Goal:** Create a deployable Ollama model with your custom system prompt.

**Requirements:**
- [ ] Write a `Modelfile` that: points to your GGUF, sets `num_ctx 4096`, sets `temperature 0.1`, sets `stop "<|im_end|>"`, and includes a system prompt specifying PostgreSQL SQL-only output
- [ ] Run `ollama create postgres-sqlcoder -f Modelfile` and verify success
- [ ] Run `ollama list` and confirm `postgres-sqlcoder` appears
- [ ] Test interactively: `ollama run postgres-sqlcoder "Write a query to count rows in the orders table"` — output must be valid SQL only, no prose
- [ ] Test via REST: `curl http://localhost:11434/v1/chat/completions -d '{"model":"postgres-sqlcoder","messages":[{"role":"user","content":"SELECT all from users"}]}'` — parse and print the SQL from the JSON response

**Deliverable:** `Modelfile` committed to your project repo + verification in `week65_results.md`

## Task 3: Build the CLI Tool

**Goal:** A polished command-line tool that any developer can use to query a PostgreSQL database schema.

**Requirements:**
- [ ] Write `sql_ask.py` with `argparse`: accepts `--schema <path>` and positional `question`
- [ ] Prompt template must match your SFT training format exactly (copy from your Week 58 dataset processing code)
- [ ] Add `--temperature` flag (default 0.1)
- [ ] Add `--model` flag (default `postgres-sqlcoder`) to allow switching models
- [ ] Handle errors gracefully: if Ollama is not running, print a helpful error message (not a Python traceback)
- [ ] Test on at least 10 questions against your TimescaleDB schema; record accuracy in `week65_results.md`
- [ ] Add a `--stream` flag that prints tokens as they arrive (use Ollama streaming API)

**Deliverable:** `sql_ask.py` committed to your project repo; `week65_results.md` with 10-question accuracy report

## Task 4: Latency Benchmarking

**Goal:** Characterize the latency of your local deployment for interactive use.

**Requirements:**
- [ ] Measure TTFT (time to first token) for 10 prompts of varying length (short schema ~200 tokens, full schema ~800 tokens)
- [ ] Measure total generation time for the same 10 prompts
- [ ] Build a simple Python latency harness (not curl — use streaming httpx to capture first chunk time)
- [ ] Present results as a table in `week65_results.md`: prompt tokens | TTFT | total time | tok/s
- [ ] Identify the prompt length at which TTFT exceeds 1 second (your "interactive limit")

**Deliverable:** Latency table in `week65_results.md`

## Stretch Goals

- Package `sql_ask.py` as a proper CLI tool with `pyproject.toml` and `pip install -e .` so it is available as `sqlask` command globally
- Add a `--execute` flag that pipes the generated SQL into `psql` and prints the result (requires a running PostgreSQL instance)
- Build an interactive REPL mode: keep the schema loaded, prompt for questions in a loop, maintain conversation history across turns
- Try `ollama run postgres-sqlcoder` in multi-line mode and test whether the model handles follow-up questions like "now add a WHERE clause for last week"
