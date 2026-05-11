# Week 67 — Technical Report Week 1: Outline, Problem Statement, Dataset Section

## Learning Objectives

By the end of this week, you will be able to:

- Structure a machine learning technical report following the conventions of published LLM papers
- Write a clear problem statement that positions your work relative to prior work
- Document your dataset construction methodology with enough detail for reproduction
- Use academic writing conventions without losing clarity
- Produce a draft abstract that accurately summarizes contributions and results

## Concepts

### What a Technical Report Is (and Is Not)

A technical report for a fine-tuned LLM is not a research paper claiming a novel theoretical contribution — your contribution is engineering: a careful, reproducible pipeline applied to a specific domain. The best models for your report are papers like the Llama 2 technical report (Touvron et al. 2023), SQLCoder's model card, and Tulu 2 (Ivison et al. 2023). These papers are clear, dense, and emphasize reproducibility over novelty.

Your report will be structured as:
1. Abstract
2. Introduction / Problem Statement
3. Dataset Construction
4. Training Pipeline
5. Evaluation
6. Ablations
7. Limitations and Future Work
8. Appendix (reproducibility, hyperparameters)

Weeks 67–70 each produce two sections plus the full document integration. This week: Abstract draft, Introduction, and Dataset Construction.

### Writing the Abstract

The abstract does four things in four sentences: (1) state the problem, (2) describe your approach, (3) report key numbers, (4) state what you release.

Example structure:

"Text-to-SQL for production PostgreSQL databases remains unsolved at open-weight model scale: existing models achieve 82% on Spider but drop below 65% on real-world TimescaleDB schemas with time-series functions. We present postgres-sqlcoder-7b, a Qwen2.5-Coder-7B model fine-tuned through a four-stage pipeline — continued pretraining on 100M PostgreSQL tokens, supervised fine-tuning on 25K curated examples, DPO on 5K preference pairs, and GRPO with executable-SQL rewards. Our model achieves 83.1% exact-match accuracy on a custom TimescaleDB benchmark, outperforming GPT-4o (79.4%) and Claude 3.5 Sonnet (81.2%) on this domain while remaining under 5 GB quantized. We release all training code, datasets, and three quantized model variants."

Write your own abstract using your actual numbers from Weeks 61–62.

### Writing the Introduction

The introduction situates your work in 3–4 paragraphs:

Paragraph 1: The problem. Text-to-SQL is industrially valuable but hard. Prior work (Spider, BIRD, Defog) benchmarks on standard schemas. Production databases — especially TimescaleDB with hypertables, continuous aggregates, and time-bucket functions — introduce patterns not covered by existing benchmarks.

Paragraph 2: The gap. Closed-source models (GPT-4o, Claude) perform reasonably but are expensive, slow (API latency), and cannot be deployed on-premise. Existing open-weight SQL models (SQLCoder-7B, DeepSeek-Coder) are not fine-tuned for TimescaleDB and do not handle multi-turn queries.

Paragraph 3: Your contributions. List them explicitly as bullets:
- A 100M-token PostgreSQL continued pretraining corpus
- A 25K-example SFT dataset combining LIMA-style curation with Magpie-style synthetic generation
- A four-stage training pipeline: CPT → SFT → DPO → GRPO
- A 200-example TimescaleDB evaluation benchmark (released publicly)
- Three quantized variants (Q4_K_M GGUF, AWQ INT4, GPTQ INT4) for local deployment

Paragraph 4: Paper overview — "Section 3 describes dataset construction; Section 4 the training pipeline..."

### Writing the Dataset Section

The dataset section is where reproducibility lives. For each dataset component, document:

- Source: where examples came from
- Size: exact number of examples
- Construction process: filtering, deduplication, quality criteria
- Format: the prompt template used
- Statistics: schema size distribution, query length distribution, SQL complexity distribution

Structure for your report:

```
3. Dataset Construction

3.1 Continued Pretraining Corpus
  - Source: PostgreSQL docs, TimescaleDB docs, pgExercises, Stack Overflow (SQL tag)
  - Size: 102M tokens after deduplication
  - Processing: byte-pair deduplication with 13-gram fingerprinting (minhash LSH)
  - Format: raw text, no chat template

3.2 SFT Dataset v3
  3.2.1 Human-curated subset (LIMA-style): 1,247 examples, hand-verified
  3.2.2 Magpie-generated subset: 18,432 examples
  3.2.3 Spider/BIRD adapted: 5,821 examples with TimescaleDB schema augmentation
  Total: 25,500 examples after LLM-as-judge filtering (Section 3.4)

3.3 DPO Preference Dataset
  - 5,000 (chosen, rejected) pairs
  - Chosen: passed LLM judge + executable on test DB
  - Rejected: syntactically valid but wrong result

3.4 Quality Filtering
  - LLM-as-judge using GPT-4o-mini, threshold: score ≥ 4/5
  - Acceptance rate: 67% of generated examples passed
  - Deduplication: removed examples with >0.85 ROUGE-L overlap
```

Include one or two example data points (schema + question + SQL triple) in the text to give readers intuition.

### Academic Writing Conventions

Use past tense for methods you applied. Describe what you did, not what "we recommend doing." Avoid hedging: instead of "we believe the filtering improved quality," write "filtered examples achieved 4.1 pp higher accuracy (Table 2)."

Cite related work inline with author-year format. The key papers to cite: Spider (Yu et al. 2018), BIRD (Li et al. 2023), SQLCoder (Defog 2023), Tulu 2 (Ivison et al. 2023), LIMA (Zhou et al. 2023), DPO (Rafailov et al. 2023), GRPO (Shao et al. 2024), Qwen2.5-Coder (Hui et al. 2024).

## Connections

This week begins the four-week writing arc (Weeks 67–70). Week 68 covers the training pipeline and architecture sections. Week 69 covers evaluation and ablations. Week 70 polishes, integrates, and publishes. The dataset numbers you document this week should match exactly what you logged in W&B during Weeks 53–58. Pull the exact numbers from your W&B dashboards, not from memory.

## Common Misconceptions / Pitfalls

A common error is writing the dataset section as a narrative ("first we did X, then Y") rather than as structured documentation. Use numbered subsections, tables, and bullet lists. Readers skip to the section they care about — make it scannable.

Do not pad the abstract with motivation sentences. Every sentence in the abstract must carry information that cannot be inferred from the others.

Do not report benchmark numbers you did not personally measure. If you could not run GPT-4o on your 200-example benchmark, say "GPT-4o on BIRD-SQL dev (published)" vs "postgres-sqlcoder-7b on BIRD-SQL dev (ours)."

## Time Allocation (6–8 hours)

- 1.0h: Read 2–3 reference technical reports (Llama 2, Tulu 2, SQLCoder model card)
- 0.5h: Outline all sections in `report_outline.md`
- 1.5h: Write abstract (3 drafts, pick best)
- 1.5h: Write introduction
- 2.0h: Write dataset section with statistics tables
- 0.5h: Review and revise for clarity and accuracy
