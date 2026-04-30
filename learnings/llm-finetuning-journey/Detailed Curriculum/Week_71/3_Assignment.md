# Week 71 Assignment — Frontier Reading: Tulu 3, SmolLM2, OLMo 2

## Setup Checklist

- [ ] Download PDFs: Tulu 3 (arXiv 2411.15124), SmolLM2 (arXiv 2502.05654), OLMo 2 (arXiv 2501.00656)
- [ ] `reading_notes/` directory created in your project repo
- [ ] Your own postgres-sqlcoder-7b technical report nearby for cross-referencing

## Task 1: Structured Paper Extraction — Tulu 3

**Goal:** Extract the three most reusable insights from Tulu 3 for your work.

**Requirements:**
- [ ] Read Abstract + Introduction: write 2 sentences summarizing the top claimed contribution
- [ ] Find and copy the main results table: identify one benchmark where Tulu 3 is strongest vs one where it underperforms
- [ ] Read the RLVR section: explain in your own words how RLVR differs from RLHF; where does your GRPO training fit on this spectrum?
- [ ] Read the data section: what is the acceptance rate of their data generation pipeline vs yours (Week 55)?
- [ ] Write 3 bullet points: "If I applied Tulu 3's X to postgres-sqlcoder-7b, I expect Y because Z"
- [ ] Save as `reading_notes/tulu3_notes.md`

**Deliverable:** `reading_notes/tulu3_notes.md` (300–500 words)

## Task 2: Structured Paper Extraction — SmolLM2

**Goal:** Extract the key data quality and curriculum insights applicable to small/mid-size SQL models.

**Requirements:**
- [ ] Find the section describing FineWeb-Edu and DCLM data curation: what quality signals do they use?
- [ ] Identify the key accuracy vs model size tradeoff result: at what parameter count does SmolLM2 match larger models and on which benchmark?
- [ ] Relate SmolLM2's tokenizer efficiency findings to your model: how many tokens does `time_bucket(ts, INTERVAL '1 hour')` take in Qwen2.5's tokenizer?
- [ ] Write 3 bullet points: "SmolLM2's finding X has implication Y for my Week 75 base model comparison because Z"
- [ ] Save as `reading_notes/smollm2_notes.md`

**Deliverable:** `reading_notes/smollm2_notes.md` (300–500 words)

## Task 3: Structured Paper Extraction — OLMo 2

**Goal:** Extract the open infrastructure and mid-training data mixing insights.

**Requirements:**
- [ ] Find the mid-training data mixing section: what data sources are upweighted and by how much?
- [ ] Find the ablation study: how much accuracy does the second-phase high-quality data contribute (vs first-phase pretraining)?
- [ ] Evaluate the claim that full openness (code + data + checkpoints) enables better science: give one specific example from OLMo 2's results that would not have been discoverable without intermediate checkpoints
- [ ] Compare OLMo 2's reproducibility appendix to yours: what is in theirs that is missing from yours?
- [ ] Save as `reading_notes/olmo2_notes.md`

**Deliverable:** `reading_notes/olmo2_notes.md` (300–500 words)

## Task 4: Cross-Paper Synthesis

**Goal:** A synthesis document that identifies the emerging consensus and its implications for your work.

**Requirements:**
- [ ] Write `reading_notes/week71_synthesis.md` with:
  - "What all three papers agree on" (4–5 bullet points)
  - "What each paper contributes that the others do not" (3 bullets, one per paper)
  - "Three techniques I will apply in Weeks 75–77" (specific, with reasoning)
  - "One claim from these papers that I am skeptical of, and why" (intellectual honesty)
- [ ] Length: 400–600 words

**Deliverable:** `reading_notes/week71_synthesis.md`

## Stretch Goals

- Run OLMo 2's OLMES evaluation harness on your postgres-sqlcoder-7b model and compare to OLMo 2 on the shared benchmarks
- Reproduce one SmolLM2 data quality filter on your own SQL dataset and measure the effect on held-out perplexity
- Write a 500-word "related work update" for your technical report incorporating Tulu 3 and OLMo 2 as additional related work
