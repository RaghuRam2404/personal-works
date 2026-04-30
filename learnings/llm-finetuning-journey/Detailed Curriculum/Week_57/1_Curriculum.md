# Week 57 — Continued Pretraining on a 100M-Token Domain Corpus

## Learning Objectives

By the end of this week, you will be able to:

- Explain why continued pretraining (CPT) before SFT can improve domain-specific performance
- Build a 100M-token PostgreSQL/TimescaleDB corpus from public sources
- Configure and run a CPT run on RunPod H100 using Unsloth
- Distinguish between CPT training objectives (causal LM) and SFT (instruction following)
- Monitor CPT for domain knowledge acquisition without catastrophic forgetting

## What is Continued Pretraining?

Continued pretraining sits between full pretraining and supervised fine-tuning. It is a causal language modeling objective (predict the next token) run on a domain-specific corpus, starting from a general pretrained model. No instruction-following format, no system prompts, no human/assistant turns — just raw domain text.

The motivation: Qwen2.5-Coder-7B has seen billions of SQL tokens during pretraining, but only a small fraction of those are PostgreSQL-specific, and almost none are TimescaleDB-specific. By continuing pretraining on a large PostgreSQL/TimescaleDB corpus, you push the model's "default prior" toward your domain before SFT teaches it the instruction format. The result is typically a 3–8 percentage point improvement on domain evaluation metrics compared to SFT alone.

**When to do CPT vs. skip it:**
- CPT helps most when your domain is niche enough to be underrepresented in the base model's pretraining
- CPT helps least when the base model already has strong domain coverage
- CPT risks catastrophic forgetting of general SQL knowledge if run for too many steps
- CPT at 100M tokens on a 7B model for ~3 hours on H100 is a low-risk, potentially high-reward step

## Concepts

### Building the 100M-Token Corpus

Your corpus should consist of raw text that teaches the model PostgreSQL and TimescaleDB knowledge — not instruction pairs, but documentation, examples, discussions.

**Source 1: PostgreSQL official documentation** (~15M tokens)
- Full PostgreSQL 16 HTML documentation: postgresql.org/docs/current
- Convert HTML to text; strip navigation, ads, repeated headers
- Include: SQL reference, function catalog, EXPLAIN documentation, index types, query planning

**Source 2: TimescaleDB documentation and blog** (~5M tokens)
- TimescaleDB docs (docs.timescale.com): all pages
- TimescaleDB blog posts (timescale.com/blog)
- Focus on: tutorial pages, SQL examples, hyperfunction reference

**Source 3: Stack Overflow SQL dumps** (~60M tokens)
- Download the Stack Overflow data dump (archive.org or official)
- Filter for posts tagged `[postgresql]` or `[timescaledb]`
- Include questions + accepted answers only (quality filter)
- Strip HTML, normalize whitespace

**Source 4: GitHub SQL files** (~15M tokens)
- From The Stack v2 (HuggingFace), filter `.sql` files
- Apply language classification: keep only files with PostgreSQL-specific keywords
- Include `.sql` migration files, schema definitions, stored procedures

**Source 5: PostgreSQL Wiki and mailing lists** (~5M tokens)
- PostgreSQL wiki (wiki.postgresql.org) — SQL anti-patterns, best practices
- PostgreSQL mailing list archives — "pgsql-sql" and "pgsql-performance" digests

**Total target:** 100M tokens. Verify with: `sum(len(tokenizer.encode(text)) for text in corpus)`

### Corpus Preprocessing

CPT corpus preprocessing differs from SFT preprocessing:

- No formatting into prompts — raw text, one document per line (or separated by EOS tokens)
- Light deduplication at document level (URL-based exact dedup)
- Quality filtering: remove pages with > 40% non-ASCII characters (broken encodings), < 50 words, or > 95% repeating character patterns
- No shuffling between source-specific "books" — many practitioners interleave sources randomly, but document-level shuffling is sufficient

**Packing documents:** To maximize GPU utilization, pack multiple documents into each training sequence. Each sequence has length `max_seq_len` (typically 2048 or 4096). Insert an EOS token between documents. Use `DataCollatorForLanguageModeling` with `mlm=False` (causal LM).

### Training Configuration for CPT

CPT is run with different hyperparameters than SFT:

| Setting | SFT | CPT |
|---------|-----|-----|
| Learning rate | 2e-4 (LoRA) | 1e-4 to 5e-5 |
| Batch size | 4–16 | 32–128 |
| LoRA rank | 16–64 | 16 (or full params) |
| Epochs | 1–3 | 1 (never more than 1) |
| Weight decay | 0.01 | 0.1 |
| Warmup | 100–200 steps | 1000 steps |

**Critical: run CPT for exactly 1 epoch.** Multi-epoch CPT on domain text causes catastrophic forgetting of general knowledge. The goal is to shift the model's distribution toward your domain, not overfit it.

**LoRA for CPT:** You can use LoRA for CPT (cheaper, less VRAM, still effective) but you need to target more modules than SFT — include attention, MLP, and embedding layers for CPT. Alternatively, some practitioners do full fine-tuning for CPT on H100 (Qwen-7B fits in H100 80GB at bf16). Full fine-tuning for CPT is generally better than LoRA but more expensive.

### Monitoring CPT

Log these metrics to W&B:
- Training loss (should decrease from ~2.5 to ~1.8 on a well-filtered domain corpus)
- Perplexity on a held-out PostgreSQL documentation page (not in training set)
- Perplexity on a held-out general English text (should not increase significantly — indicates forgetting)
- TimescaleDB-specific token prediction accuracy (track specific token sequences like "time_bucket" completion)

**Early stopping for CPT:** If general text perplexity increases by more than 0.5 bits, stop training. Domain specialization is not worth catastrophic forgetting.

### Compute: RunPod H100

The 100M-token corpus at batch size 64, sequence length 2048, for 1 epoch:
- Total tokens: 100M
- Tokens per step: 64 × 2048 = 131,072
- Total steps: 100M / 131,072 ≈ 763 steps
- Time at H100 (~100B tokens/hour with Unsloth): ~1M tokens/minute → ~100 minutes
- Cost: H100 80GB at ~$2.79/hr → 763 steps takes ~1.5–2 hours → ~$4–6

This fits comfortably in your budget.

### Common Misconceptions and Pitfalls

**"I should do CPT for multiple epochs."** Do not. The model has already seen similar text during its initial pretraining. One epoch is enough to shift the distribution. More epochs cause forgetting.

**"CPT and SFT are the same thing."** Fundamentally different objectives: CPT predicts next token on raw text (no labels beyond the text itself); SFT predicts completions to instruction prompts. The data format, training objective, and learning rate differ.

**"I need 1B tokens for CPT to matter."** 100M tokens is enough to produce measurable improvement on domain eval. The returns diminish steeply after 500M tokens for a 7B model.

## Connections

This week's output (a CPT checkpoint) is the starting point for Week 58's SFT run. The CPT checkpoint should be saved to HuggingFace as `<your-handle>/qwen2.5-coder-7b-postgres-cpt`. Week 68's technical report section covers the CPT step.

## Time Allocation (6–8 hrs)

- 2h: Build the 100M-token corpus (download, filter, tokenize, verify token count)
- 1h: Configure and test the training script on a 100-step smoke test (Colab free, no RunPod yet)
- 0.5h: Spin up RunPod H100 instance, upload corpus
- 2h: Run CPT on RunPod (~1.5hr training + monitoring)
- 0.5h: Evaluate CPT checkpoint on held-out domain perplexity; compare to base model
- 0.5h: Push checkpoint to HuggingFace; commit code; log to W&B
- 0.5h: Shut down RunPod instance (critical — do not leave it running)
