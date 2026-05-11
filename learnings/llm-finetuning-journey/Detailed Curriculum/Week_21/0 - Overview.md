# Week 21 Overview — Run 50M LM Pretraining

This file is your map for Week 21. Read it first; everything else fits inside it.

## The story this week

This is the most hands-on week of Phase 3. You are running a real training job — not a tutorial, not a toy, but a full pretraining run targeting ~2 billion tokens across approximately 24 hours of A100 GPU time (spread across the week).

## What you need to do

- [ ] `train.bin` contains at least 500M tokens (1B preferred)
- [ ] 200-step sanity check passes: initial loss ~10.4, loss after 200 steps < 7.0
- [ ] W&B project `week-21-50m-pretrain` created and verified (run 1 step, check W&B receives the data)
- [ ] Checkpoint directory exists and write permissions verified
- [ ] Accelerate config set to bf16 mixed precision on single GPU
- [ ] `torch.compile(model)` is applied if using PyTorch 2.0+ (adds ~5 min compile time, then faster)
- [ ] `train.py` has `--resume` argument that loads the latest checkpoint in `checkpoints/`

Concretely, by the end of the week you should be able to:

- Execute a multi-hour pretraining run with proper monitoring and checkpointing
- Identify divergence, instability, and healthy training from W&B loss curves
- Debug common training failure modes (loss spike, NaN loss, OOM) under time pressure
- Interpret tokens/sec throughput and estimate remaining training time
- Recover from training interruptions by resuming from checkpoints

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

- 1h: Pre-launch checklist (re-run sanity check from Week 20, verify checkpointing works)
- 5–6h: Active training on Colab A100 (can be background — check W&B every 30 min)
- 1h: Debug any issues, log findings in `journal.md`
- 0.5h: Commit checkpoint link and journal entry

## Why this week matters

**One-liner:** Plan the token budget, checkpoint every 2000 steps, read loss curves critically, never change hyperparameters mid-run unless catastrophic.
