# Week 65 Overview — Local Deployment: Ollama, llama.cpp, and a CLI SQL Tool

This file is your map for Week 65. Read it first; everything else fits inside it.

## The story this week

llama.cpp compiles to native Metal kernels on macOS, offloading matrix multiplications to the GPU on Apple Silicon. The critical build flag is `-DLLAMA_METAL=ON`. Without it, inference runs on CPU cores and is 3–5x slower.

## What you need to do

- [ ] `postgres-sqlcoder-7b-Q4_K_M.gguf` available locally (from Week 64)
- [ ] llama.cpp cloned and built with Metal: `cmake -B build -DLLAMA_METAL=ON && cmake --build build -j8`
- [ ] Ollama installed: `brew install ollama` or download from `https://ollama.com`
- [ ] Python environment with `httpx` installed: `pip install httpx`
- [ ] A PostgreSQL schema file for testing (your TimescaleDB orders schema from earlier weeks, or create a simple one)

Concretely, by the end of the week you should be able to:

- Build llama.cpp from source on Apple Silicon with Metal acceleration enabled
- Register your Q4_K_M GGUF as an Ollama model with a custom system prompt
- Serve your model through Ollama's OpenAI-compatible REST API
- Write a CLI tool in Python that accepts a PostgreSQL schema and natural-language question and returns executable SQL
- Benchmark local latency end-to-end from prompt to first token

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

- 1.0h: Build llama.cpp from source, verify Metal inference
- 1.0h: Install Ollama, write Modelfile, test with ollama run
- 1.5h: Write and test sql_ask.py CLI tool
- 1.0h: Latency profiling — TTFT and total generation time
- 1.0h: Accuracy spot-check on 20 custom examples through the CLI
- 0.5h: Clean up, write README for the tool, commit to your project repo

## Why this week matters

Your model runs locally at 40–55 tok/s on Apple Silicon — fast enough for interactive SQL generation from any terminal.
