# Week 13 Overview — KV Cache, Inference Optimization, and Sampling

This file is your map for Week 13. Read it first; everything else fits inside it.

## The story this week

During training, you feed a full sequence of length T, compute all T positions in parallel (enabled by the causal mask), and backpropagate. The entire sequence is processed in one GPU kernel call.

## What you need to do

- [ ] Your Week 12 `model_v2.py` (modernized nanoGPT) or Week 11 `model.py` as base
- [ ] A trained checkpoint from Week 11 or 12 (for inference benchmarking)
- [ ] GitHub branch `week-13-kv-cache-sampling`
- [ ] W&B project `week-13-kv-cache-sampling`
- [ ] Colab Free or Mac (inference runs fast on CPU for this model size)

Concretely, by the end of the week you should be able to:

- Explain exactly what the KV cache stores and why it eliminates redundant computation at inference time
- Implement a working KV cache from scratch in your nanoGPT
- Implement temperature, top-k, and top-p (nucleus) sampling from scratch
- Benchmark inference speed with and without KV cache and explain the speedup
- Describe the trade-offs between greedy, top-k, top-p, and temperature sampling
- Explain why beam search is rarely used in modern LLM inference

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
| Read The Illustrated GPT-2 (KV cache section) | 0.75 hrs |
| Read HuggingFace "How to generate text" blog post | 0.5 hrs |
| Read HF Generation strategies docs | 0.5 hrs |
| Implement KV cache in nanoGPT | 2.5 hrs |
| Implement all sampling strategies, benchmark | 2.5 hrs |
| Write commit + notes | 0.75 hrs |

---

## Why this week matters

**This week in 15 words:** Cache K and V from past tokens; sample smartly — temperature, top-k, top-p; never recompute.
