# Week 66 Overview — Cloud Deployment: vLLM, OpenAI-Compatible API, HF Spaces

This file is your map for Week 66. Read it first; everything else fits inside it.

## The story this week

vLLM is the production-grade inference server for LLMs on GPU. Its two core innovations are paged attention (managing KV cache as virtual memory pages to eliminate fragmentation) and continuous batching (dynamically adding new requests to an ongoing batch mid-inference, rather than waiting for all sequences to finish).

## What you need to do

- [ ] RunPod account with billing enabled (A10G pod costs ~$0.50–0.75/hr; budget $5 for this week)
- [ ] AWQ INT4 model pushed to HuggingFace Hub in Week 64: `<your-handle>/postgres-sqlcoder-7b-awq-int4`
- [ ] `HF_TOKEN` environment variable set (for private repos)
- [ ] Python packages on RunPod: `vllm`, `fastapi`, `uvicorn`, `openai`, `sqlparse`
- [ ] Local: `openai` Python client installed

Concretely, by the end of the week you should be able to:

- Launch a vLLM server on a RunPod GPU instance serving your AWQ INT4 model
- Configure vLLM for continuous batching and paged attention for high-throughput inference
- Test the deployment through the OpenAI-compatible `/v1/chat/completions` endpoint
- Build a minimal FastAPI wrapper that adds schema injection and SQL post-processing
- Optionally deploy a Gradio demo to Hugging Face Spaces

## Suggested order through the files

1. **1 - Curriculum.md** — read first. Concepts, connections, common pitfalls.
2. **2 - Resources.md** — papers, videos, blog posts, repos, docs to consult while studying.
3. **3 - Assignment.md** — the hands-on task you must complete this week.
4. **4 - AssignmentSolutions.md** — only after you have attempted the assignment yourself.
5. **5 - Quiz.md** — self-test that you have absorbed the week's material.
6. **6 - Answers.md** — quiz answers and explanations; check after attempting.
7. **7 - Glossary.md** — terminology used this week, indexed for fast lookup.
8. **8 - TakeAway.md** — the one-page summary you keep for the rest of the course.

## Time budget

- 0.5h: Provision RunPod A10G instance
- 1.0h: Install vLLM, download model, launch server, verify endpoint
- 1.0h: Test throughput — concurrent requests, measure tok/s vs single-request baseline
- 1.5h: Build FastAPI wrapper with schema injection and SQL cleaning
- 1.0h: Write integration test: 20 questions through the API, check accuracy
- 1.0h: Optional — HF Spaces Gradio demo with GGUF via llama-cpp-python
- 0.5h: Document API, write README, add costs to `week66_results.md`

## Why this week matters

vLLM + paged attention + continuous batching turns one A10G into an SQL API that handles 120+ queries/minute at sub-second latency.
