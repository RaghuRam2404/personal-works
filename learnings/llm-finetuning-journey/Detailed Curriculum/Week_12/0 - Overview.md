# Week 12 Overview — Modern Architectural Improvements (RMSNorm, SwiGLU, RoPE, GQA, SWA)

This file is your map for Week 12. Read it first; everything else fits inside it.

## The story this week

The original Transformer from Week 10 uses Post-LN, learned absolute positional embeddings, ReLU activations, and full multi-head attention where every head reads/writes the full KV cache. Between 2019 and 2024, each of these choices was improved. These improvements are not cosmetic — they are what separate a research prototype from a production LLM. LLaMA, Mistral, Qwen, and Gemma all use exactly this combination.

## What you need to do

- [ ] Your nanoGPT from Week 11 (`model.py`) as the base
- [ ] Create a new file `model_v2.py` — do not overwrite your Week 11 model
- [ ] GitHub branch `week-12-modern-arch`
- [ ] W&B project `week-12-modern-arch` with two runs: `baseline` and `modernized`
- [ ] Colab Free (or Mac MPS — this is small enough)

Concretely, by the end of the week you should be able to:

- Implement RMSNorm and explain why it drops the re-centering step of LayerNorm
- Implement SwiGLU and explain why gated activations improve FFN expressiveness
- Implement Rotary Position Embeddings (RoPE) using complex number rotation
- Explain Grouped-Query Attention (GQA) and compute its KV head memory savings
- Modify your nanoGPT to replace all four components and compare val loss
- Describe Sliding Window Attention and when it is appropriate

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
| Read papers (Pre-LN/RMSNorm/SwiGLU — ~15p total) | 1 hr |
| Read RoPE paper (2104.09864) — sections 1–3 | 1 hr |
| Read GQA paper (2305.13245) | 0.5 hrs |
| Watch Umar Jamil LLaMA video (1h10m) | 1.25 hrs |
| Implement and integrate all 4 improvements into nanoGPT | 3.5 hrs |
| Compare val loss, commit | 0.75 hrs |

---

## Why this week matters

**This week in 15 words:** RMSNorm, SwiGLU, RoPE, GQA — the four changes that turn GPT into LLaMA.
