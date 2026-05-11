# Week 14 Overview — Reading and Understanding the LLaMA Papers

This file is your map for Week 14. Read it first; everything else fits inside it.

## The story this week

LLaMA (Large Language Model Meta AI), released by Meta AI in February 2023, was the first high-quality open-weights LLM family. Before LLaMA, the dominant models were GPT-3, PaLM, Chinchilla — all closed weights, API-only. LLaMA changed the landscape: researchers could now fine-tune, quantize, and modify a competitive model on consumer hardware.

## What you need to do

This is a reading and annotation week. There is no new model training. The deliverables are written notes and annotations that demonstrate you read the papers carefully.

Concretely, by the end of the week you should be able to:

- Describe the key design decisions in LLaMA 1 and explain why each was made
- Compare the architectural and training differences across LLaMA 1, 2, and 3
- Read and annotate `modeling_llama.py` from the HuggingFace transformers library
- Explain `num_key_value_heads` in the HF implementation and how it implements GQA
- Write a 1-page summary mapping `modeling_llama.py` to your Week 12 nanoGPT modernization
- Identify where RoPE, RMSNorm, SwiGLU, and GQA appear in production code

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
| Read LLaMA 1 paper fully (2302.13971) | 2 hrs |
| Read LLaMA 2 paper — arch changes section (2307.09288) | 0.75 hrs |
| Read LLaMA 3 paper — Sections 1, 3, 5 (2407.21783) | 1 hr |
| Print and annotate `modeling_llama.py` line by line | 2.5 hrs |
| Write `journal.md` summary | 0.75 hrs |

This is a reading-heavy week. There is no new coding. The coding deliverable is annotation and written notes.

---

## Why this week matters

**This week in 15 words:** LLaMA is your Week 12 nanoGPT, engineered at scale with RMSNorm, SwiGLU, RoPE, and GQA.
