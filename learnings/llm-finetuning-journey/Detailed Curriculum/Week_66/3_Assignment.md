# Week 66 Assignment — Cloud Deployment with vLLM

## Setup Checklist

- [ ] RunPod account with billing enabled (A10G pod costs ~$0.50–0.75/hr; budget $5 for this week)
- [ ] AWQ INT4 model pushed to HuggingFace Hub in Week 64: `<your-handle>/postgres-sqlcoder-7b-awq-int4`
- [ ] `HF_TOKEN` environment variable set (for private repos)
- [ ] Python packages on RunPod: `vllm`, `fastapi`, `uvicorn`, `openai`, `sqlparse`
- [ ] Local: `openai` Python client installed

## Task 1: Launch vLLM on RunPod

**Goal:** A running vLLM endpoint serving your AWQ model on a cloud GPU.

**Requirements:**
- [ ] Launch a RunPod pod with A10G (24 GB) or RTX 3090 (24 GB), RunPod PyTorch template
- [ ] Expose TCP port 8000 in pod network settings
- [ ] Install vLLM and launch: `python -m vllm.entrypoints.openai.api_server --model <handle>/postgres-sqlcoder-7b-awq-int4 --quantization awq --served-model-name postgres-sqlcoder --max-model-len 4096 --port 8000`
- [ ] Confirm the server is running: `curl http://<pod-ip>:8000/health` returns `{"status":"ok"}`
- [ ] Run a smoke test via the OpenAI Python client — generate one SQL query successfully
- [ ] Measure and record single-request throughput (tok/s) in `week66_results.md`

**Deliverable:** `week66_results.md` with pod specs, launch command, and single-request benchmark

## Task 2: Throughput Testing

**Goal:** Understand vLLM's continuous batching advantage over sequential inference.

**Requirements:**
- [ ] Write a Python script that sends N concurrent requests using `asyncio` + `aiohttp` or `httpx.AsyncClient`
- [ ] Test with N = 1, 4, 8, 16 concurrent requests (all sending the same schema + different questions)
- [ ] Measure: total time for all N requests, average latency per request, aggregate tok/s
- [ ] At N=1: aggregate tok/s should match single-request baseline
- [ ] At N=8 or N=16: aggregate tok/s should be significantly higher than N×single-request due to batching
- [ ] Present as a table in `week66_results.md`: N | total_time | avg_latency | agg_tok/s

**Deliverable:** `load_test.py` committed + throughput table in `week66_results.md`

**Hints:** True continuous batching benefits appear at N ≥ 4. If you see linear scaling (no batching benefit), check that your requests are truly concurrent and not sequential.

## Task 3: FastAPI Wrapper

**Goal:** A production-ready SQL API that hides vLLM complexity from callers.

**Requirements:**
- [ ] Write `app.py` with FastAPI, loading the schema from `schema.sql` at startup
- [ ] `POST /sql` endpoint accepts `{"question": str, "temperature": float}`
- [ ] Response: `{"sql": str, "tokens_used": int}`
- [ ] SQL cleaning: strip markdown fences, strip trailing semicolons if present, return a single statement
- [ ] Add `GET /health` endpoint that returns `{"status": "ok", "model": "postgres-sqlcoder"}`
- [ ] Test: run 20 questions from your benchmark through the FastAPI wrapper and verify clean SQL output
- [ ] Log each request: question, generated SQL, tokens used, latency (to stdout is fine)

**Deliverable:** `app.py` committed to project repo + 20-question test results in `week66_results.md`

## Task 4: Cost Analysis

**Goal:** Understand the economics of cloud LLM inference.

**Requirements:**
- [ ] Record the total RunPod cost for this week's experiments (from the billing dashboard)
- [ ] Calculate cost per SQL query at your measured throughput (RunPod $/hr ÷ queries/hr)
- [ ] Compare to OpenAI GPT-4o API cost for the same 20 queries (use published pricing, no actual API call needed)
- [ ] Write a 100-word summary in `week66_results.md`: at what query volume does self-hosting become cheaper than GPT-4o API?

**Deliverable:** Cost analysis section in `week66_results.md`

## Stretch Goals

- Set up a HuggingFace Space with a Gradio demo using llama-cpp-python and your Q4_K_M GGUF (CPU inference, ~8–12 tok/s, but free and public)
- Add streaming support to the FastAPI wrapper using Server-Sent Events
- Add an `--execute` flag that takes a `DATABASE_URL` and runs the SQL, returning both the SQL and the query result as JSON
- Implement rate limiting in the FastAPI wrapper using `slowapi` (10 requests/minute per IP)
