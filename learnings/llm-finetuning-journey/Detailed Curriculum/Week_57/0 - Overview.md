# Week 57 Overview — Continued Pretraining on a 100M-Token Domain Corpus

This file is your map for Week 57. Read it first; everything else fits inside it.

## The story this week

Your corpus should consist of raw text that teaches the model PostgreSQL and TimescaleDB knowledge — not instruction pairs, but documentation, examples, discussions.

## What you need to do

- [ ] RunPod account with billing set up; H100 80GB instance available
- [ ] HuggingFace account with write access
- [ ] Sufficient storage on RunPod for: model weights (14GB) + corpus (JSONL ~2GB) + optimizer states (~28GB)
- [ ] Unsloth installed: `pip install unsloth` (RunPod Unsloth template preferred)
- [ ] W&B account; API key set in environment

Concretely, by the end of the week you should be able to:

- Explain why continued pretraining (CPT) before SFT can improve domain-specific performance
- Build a 100M-token PostgreSQL/TimescaleDB corpus from public sources
- Configure and run a CPT run on RunPod H100 using Unsloth
- Distinguish between CPT training objectives (causal LM) and SFT (instruction following)
- Monitor CPT for domain knowledge acquisition without catastrophic forgetting

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

- 2h: Build the 100M-token corpus (download, filter, tokenize, verify token count)
- 1h: Configure and test the training script on a 100-step smoke test (Colab free, no RunPod yet)
- 0.5h: Spin up RunPod H100 instance, upload corpus
- 2h: Run CPT on RunPod (~1.5hr training + monitoring)
- 0.5h: Evaluate CPT checkpoint on held-out domain perplexity; compare to base model
- 0.5h: Push checkpoint to HuggingFace; commit code; log to W&B
- 0.5h: Shut down RunPod instance (critical — do not leave it running)

## Why this week matters

**One-liner:** CPT = causal LM on raw domain text, exactly 1 epoch, with EOS between documents.
