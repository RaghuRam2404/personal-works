# Week 67 Assignment Solutions

## Task 1: Outline Structure

A strong outline makes the difference between a coherent report and a collection of text blocks. Here is the full structure to follow:

```markdown
# postgres-sqlcoder-7b: Technical Report

## 1. Abstract [150–200w]

## 2. Introduction [400–600w]
2.1 Problem: text-to-SQL for production PostgreSQL
2.2 Gap: closed-source cost vs open-weight domain gap
2.3 Contributions (bulleted list)
2.4 Paper overview

## 3. Dataset Construction [600–900w]
3.1 Continued pretraining corpus (102M tokens)
3.2 SFT Dataset v3 (25,500 examples)
  3.2.1 Human-curated subset (1,247 ex)
  3.2.2 Magpie-generated subset (18,432 ex)
  3.2.3 Spider/BIRD adapted (5,821 ex)
3.3 DPO preference dataset (5,000 pairs)
3.4 Quality filtering (LLM-as-judge, dedup)
Table 1: Dataset statistics summary

## 4. Training Pipeline [600–900w]  ← Week 68
## 5. Evaluation [600–900w]         ← Week 69
## 6. Ablations [300–500w]          ← Week 69
## 7. Limitations and Future Work [200–400w] ← Week 70
## 8. Appendix [300–500w]           ← Week 70
  A. Hyperparameters
  B. Compute budget
  C. Reproducibility checklist
```

## Task 2: Abstract — Template and Anti-Patterns

```markdown
# Abstract Draft 3 (Final)

Text-to-SQL for production PostgreSQL databases presents challenges not
captured by standard benchmarks: TimescaleDB schemas with time-bucket
functions, hypertables, and continuous aggregates require domain adaptation
beyond Spider or BIRD. We present postgres-sqlcoder-7b, a 7B parameter
model fine-tuned from Qwen2.5-Coder-7B through a four-stage pipeline:
100M-token continued pretraining on PostgreSQL documentation and community
content, supervised fine-tuning on 25,500 curated examples, direct
preference optimization on 5,000 preference pairs, and group relative
policy optimization with executable-SQL rewards. On our 200-example
TimescaleDB benchmark, the model achieves 83.1% exact-match accuracy,
outperforming GPT-4o (79.4%) and Claude 3.5 Sonnet (81.2%). We release
training code, datasets, and three quantized variants (Q4_K_M GGUF,
AWQ INT4, GPTQ INT4) under permissive licenses at [Hub link].
```

Anti-patterns to avoid:
- "In this paper, we propose..." — cut to the claim directly
- "...which is a challenging problem" — assume reader knows this
- Reporting only relative improvements without baseline numbers

## Task 3: Introduction — Key Structural Points

The hardest part is the contributions list. Make each bullet a claim with evidence, not a vague description:

```markdown
Our contributions are:
- **TimescaleDB benchmark**: 200 expert-written question-SQL pairs covering
  time-bucket aggregations, hypertable metadata queries, and continuous
  aggregate refresh patterns; released publicly.
- **PostgreSQL CPT corpus**: 102M tokens of domain text (docs, exercises,
  Stack Overflow) used to close the knowledge gap in Qwen2.5-Coder-7B.
- **SFT dataset v3**: 25,500 examples combining LIMA-style curation,
  Magpie-style generation, and Spider/BIRD adaptation.
- **Four-stage training recipe**: CPT → SFT → DPO → GRPO, each stage
  measured by held-out accuracy (ablation in Section 6).
- **Quantized deployment**: Q4_K_M GGUF (4.5 GB), AWQ INT4 (4.2 GB),
  GPTQ INT4 (4.5 GB) with throughput benchmarks.
```

## Task 4: Dataset Table Format

```markdown
| Component | Source | Examples/Tokens | Filtering | Format |
|-----------|--------|-----------------|-----------|--------|
| CPT corpus | PG docs + TimescaleDB + pgExercises + SO | 102M tokens | MinHash dedup (0.13-gram) | Raw text |
| SFT curated | Manual curation | 1,247 | Human review | Chat template |
| SFT synthetic | Magpie (GPT-4o-mini) | 18,432 | LLM judge ≥4/5 | Chat template |
| SFT adapted | Spider 1.0 + BIRD-SQL | 5,821 | Schema augmented + filtered | Chat template |
| DPO pairs | SFT model outputs | 5,000 | Executable chosen, wrong rejected | Prompt + chosen + rejected |
```

## Common Gotchas

- Do not write in the future tense ("we will show") — use past tense for what you did, present for what the paper describes.
- Keep exact version numbers for every dataset: "Spider 1.0" not "Spider"; "BIRD-SQL dev set (1,534 examples)" not "the BIRD benchmark."
- Your dataset section word count will balloon if you describe filtering with no quantification. Attach percentages to every filter step.
- One data example in the paper is worth two paragraphs of description — show a real schema snippet + question + SQL triple.

## How to Verify You Did It Right

Ask: can a new researcher reproduce your dataset from Section 3 alone, without reading your code? If the answer is no, add the missing details. The test is whether the section would pass a reproducibility review at an NLP venue.
