# Week 74 Overview — Frontier Reading 4: Context Extension — LongRoPE and YaRN

This file is your map for Week 74. Read it first; everything else fits inside it.

## The story this week

Your current model uses a 4096-token context window. A typical SQL prompt contains:
- System prompt: ~50 tokens
- Database schema: 200–1500 tokens (depends on table count and column count)
- Question: 20–80 tokens
- Generated SQL: 50–300 tokens

## What you need to do

- [ ] Download PDFs: YaRN (arXiv 2309.00071), LongRoPE (arXiv 2402.13753)
- [ ] Your merged BF16 model or a small proxy model available
- [ ] A long schema prompt prepared: at least one PostgreSQL schema with 15+ tables (concatenate several of your training schemas if needed, target 5000–6000 tokens)
- [ ] Flash Attention 2 installed: `pip install flash-attn --no-build-isolation`

Concretely, by the end of the week you should be able to:

- Explain why RoPE (Rotary Position Embedding) limits context length and why naive context extension fails
- Describe how YaRN extends context via Non-Uniform interpolation and NTK-aware scaling
- Describe how LongRoPE uses evolutionary search to find non-uniform rescaling factors
- Assess whether context extension is relevant to your PostgreSQL SQL use case
- Identify when longer context would and would not improve your model's SQL accuracy

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

- 1.5h: Read YaRN paper (arXiv 2309.00071) — focus on Method section (Sections 3–4)
- 1.5h: Read LongRoPE paper (arXiv 2402.13753) — focus on Method section and ablations
- 1.0h: Read "RoPE" original paper (arXiv 2104.09864) if RoPE is unfamiliar — Sections 1–3
- 1.5h: Implement YaRN rope_scaling config for Qwen2.5; test perplexity on a 6000-token SQL schema prompt
- 0.5h: Write synthesis notes in `reading_notes/week74_synthesis.md`

## Why this week matters

RoPE breaks at long contexts because rotation angles go out-of-distribution; YaRN fixes this cheaply (400 steps); LongRoPE fixes it optimally (100+ GPU-hours).
