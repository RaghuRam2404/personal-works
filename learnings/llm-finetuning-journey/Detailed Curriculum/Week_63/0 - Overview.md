# Week 63 Overview — Quantization Deep Dive Part 1: GGUF, GPTQ, AWQ Comparison

This file is your map for Week 63. Read it first; everything else fits inside it.

## The story this week

For transformer inference (not training), the bottleneck is almost never FLOPS — it is the bandwidth required to load model weights from memory into the compute units for each token generation step. At 7B parameters × 2 bytes (bf16) = 14GB, each auto-regressive step requires loading up to 14GB of weights. At typical GPU memory bandwidth (900 GB/s for H100, 300 GB/s for A100, 68 GB/s for M1 Mac), this determines the speed ceiling.

## What you need to do

- [ ] `llama.cpp` cloned and built: `git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && make`
- [ ] `auto-gptq` installed: `pip install auto-gptq optimum`
- [ ] `autoawq` installed: `pip install autoawq`
- [ ] A reference 7B model (use `Qwen/Qwen2.5-Coder-7B-Instruct` — not your final model, which you quantize in Week 64)
- [ ] 100 BIRD-SQL dev examples for comparison eval

Concretely, by the end of the week you should be able to:

- Explain the mathematical basis of post-training quantization (PTQ) for LLMs
- Describe the key differences between GGUF/llama.cpp, GPTQ, and AWQ quantization methods
- Choose the appropriate quantization format for each deployment scenario
- Understand the accuracy-throughput-memory trade-off at each quantization level
- Read a quantization comparison table and interpret what the numbers mean for your use case

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

- 2h: Read and take notes on GPTQ paper abstract + Section 2–3 (methodology)
- 2h: Read AWQ paper abstract + Section 3 (methodology)
- 1h: Study the GGUF K-quants documentation in the llama.cpp repository
- 2h: Create the comparison study: run a reference 7B model (e.g., the base Qwen) through all three quantization methods on 100 BIRD-SQL questions; compare accuracy and speed
- 1h: Document findings in `quantization_comparison_study.md`

## Why this week matters

**One-liner:** Q4_K_M for Mac/CPU deployment; AWQ INT4 for GPU cloud serving; GPTQ INT4 as alternative GPU format.
