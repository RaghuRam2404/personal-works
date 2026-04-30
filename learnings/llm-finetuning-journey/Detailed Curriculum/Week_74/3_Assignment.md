# Week 74 Assignment — Frontier Reading: LongRoPE and YaRN

## Setup Checklist

- [ ] Download PDFs: YaRN (arXiv 2309.00071), LongRoPE (arXiv 2402.13753)
- [ ] Your merged BF16 model or a small proxy model available
- [ ] A long schema prompt prepared: at least one PostgreSQL schema with 15+ tables (concatenate several of your training schemas if needed, target 5000–6000 tokens)
- [ ] Flash Attention 2 installed: `pip install flash-attn --no-build-isolation`

## Task 1: YaRN Paper Extraction

**Goal:** Understand YaRN well enough to implement it for Qwen2.5.

**Requirements:**
- [ ] Explain NTK-aware scaling in your own words (2–3 sentences): what is NTK, what does "aware" mean here, and what problem does it solve?
- [ ] Find the YaRN attention temperature formula: what is the scaling factor applied to the attention logits and why?
- [ ] Find the results table: at what context length does naive position interpolation break down (perplexity > 1.5x baseline) for LLaMA?
- [ ] Find the fine-tuning requirement: how many steps are needed for YaRN to converge at 2x context extension?
- [ ] Save as `reading_notes/yarn_notes.md`

**Deliverable:** `reading_notes/yarn_notes.md` (300–400 words)

## Task 2: LongRoPE Paper Extraction

**Goal:** Understand how LongRoPE's evolutionary search improves on YaRN.

**Requirements:**
- [ ] Explain the evolutionary search algorithm in your own words: what is being searched, what is the objective function, how many generations?
- [ ] Find the comparison table: at 128K context, how much better is LongRoPE vs YaRN in perplexity (exact numbers)?
- [ ] Find the two-stage training strategy: what is each stage and why does it prevent perplexity degradation at short contexts?
- [ ] Assess: given a budget of 2 GPU-hours for context extension, which method (YaRN or LongRoPE) is more practical and why?
- [ ] Save as `reading_notes/longrope_notes.md`

**Deliverable:** `reading_notes/longrope_notes.md` (300–400 words)

## Task 3: Context Length Stress Test

**Goal:** Understand where your model's current 4096-token context actually breaks.

**Requirements:**
- [ ] Construct a long SQL prompt: concatenate 3–4 of your largest training schemas to reach 5000–6000 tokens
- [ ] Measure current model accuracy on this truncated prompt vs the same question with schema truncated to 2048 tokens
- [ ] Apply YaRN rope_scaling config to your model (no fine-tuning yet) and test on the 5000-token prompt — record perplexity and SQL accuracy
- [ ] Record results in `reading_notes/context_extension_results.md`:
  - Prompt length (tokens) | truncated-4096 accuracy | yarn-8192 accuracy (no fine-tune) | perplexity
- [ ] Assess: is context extension necessary for your current deployment use case? At what query volume would it matter?

**Deliverable:** `reading_notes/context_extension_results.md`

## Task 4: Synthesis and Decision

**Goal:** Decide whether to apply YaRN fine-tuning to postgres-sqlcoder-7b as part of Week 75 or 76 work.

**Requirements:**
- [ ] Write `reading_notes/week74_synthesis.md` (300–400 words):
  - What problems does context extension solve for SQL generation?
  - At what schema complexity (table count / column count) does 4096 tokens become a bottleneck for your use case?
  - Would extending to 8192 tokens with YaRN (400 steps, ~0.5 GPU-hours) meaningfully improve your Custom-200 benchmark or your deployment use case?
  - Recommendation: yes or no to YaRN fine-tuning, with justification

**Deliverable:** `reading_notes/week74_synthesis.md`

## Stretch Goals

- Run the full YaRN fine-tuning (400 steps at 8192 context length on your long-schema SQL examples) and measure accuracy improvement on queries requiring schemas > 4000 tokens
- Implement a schema compression preprocessing step: instead of context extension, automatically select only the relevant tables from a large schema based on the question — compare accuracy to full-schema + YaRN
- Read Section 5 of LongRoPE (ablations) and reproduce the finding that non-uniform per-dimension scaling outperforms uniform scaling on your SQL perplexity benchmark
