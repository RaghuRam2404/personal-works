# Week 32 Overview — Quantization Fundamentals

This file is your map for Week 32. Read it first; everything else fits inside it.

## The story this week

A 7B model in FP32 requires 7 × 4 = 28GB. In FP16/BF16: 14GB. In INT8: 7GB. In 4-bit: 3.5GB. Quantization is what makes inference (and with QLoRA, training) possible on consumer hardware.

## What you need to do

- [ ] Colab Pro (T4 or A100 — T4 is sufficient)
- [ ] `pip install bitsandbytes transformers auto-gptq autoawq`
- [ ] Note: bitsandbytes requires a CUDA GPU — cannot run on Mac MPS
- [ ] Model: `Qwen/Qwen2.5-Coder-1.5B` (fast to load and quantize for benchmarking)
- [ ] Calibration dataset for GPTQ: any 100–500 SQL examples from your dataset

Concretely, by the end of the week you should be able to:

- Explain the difference between FP32, FP16, BF16, INT8, INT4, NF4, and FP4 formats and their trade-offs
- Distinguish weight-only quantization from activation quantization
- Explain how GPTQ, AWQ, and LLM.int8() work at a conceptual level
- Quantize `Qwen2.5-Coder-1.5B` to 4-bit using `bitsandbytes` and measure the memory and quality trade-offs
- Build and present a comparison table: model size, VRAM, perplexity, inference speed across quantization formats

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

| Activity | Time |
|---|---|
| Read LLM.int8() paper abstract + key sections | 1h |
| Read GPTQ and AWQ paper abstracts + sections 3–4 | 1.5h |
| Read Maarten Grootendorst's visual guide to quantization | 30m |
| Set up bitsandbytes, quantize Qwen2.5-Coder-1.5B to 4-bit | 1.5h |
| Measure model size, VRAM, inference speed; build comparison table | 1.5h |
| Commit results to GitHub | 30m |

## Why this week matters

**One-liner:** NF4 = normally-distributed-optimal 4-bit; QLoRA training uses frozen NF4 base + trainable LoRA; deployment uses GPTQ/AWQ.
