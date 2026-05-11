# Week 38 Overview — Domain-Tuning Sprint Week 2: QLoRA Fine-Tune on 15K

This file is your map for Week 38. Read it first; everything else fits inside it.

## The story this week

Weeks 29–37 were practice runs and preparation. Week 38 is where everything comes together:

## What you need to do

- [ ] Colab Pro A100 runtime selected
- [ ] `train_15k.jsonl` and `val_500.jsonl` from Week 37 uploaded to Colab or accessible via HuggingFace dataset
- [ ] `held_out_test.json` (100 examples from Week 32) accessible
- [ ] Unsloth, peft, trl, bitsandbytes installed (reuse Week 34 environment)
- [ ] W&B project `week-38-qlora-15k` created
- [ ] HuggingFace write token set

Concretely, by the end of the week you should be able to:

- Execute a full production-quality QLoRA fine-tuning run of Qwen2.5-Coder-7B on 15K SQL examples
- Apply best-practice hyperparameters from your Week 35 sweep and adapter choice from Week 36
- Monitor training with W&B and make real-time decisions if issues arise
- Compare fine-tuned model quality against the base model using your held-out test set
- Push the production adapter to HuggingFace Hub with a proper model card

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
| Verify dataset from Week 37 is correctly formatted | 30m |
| Set up and run the training script on A100 | 1h |
| Monitor training in real time via W&B | 1h |
| Run held-out evaluation (base vs. fine-tuned comparison) | 1.5h |
| Write model card and push to HuggingFace | 1h |
| Commit everything to GitHub with commit: `week-38-qlora-15k` | 30m |
| Write `week38_results.md` with findings | 1h |

## Why this week matters

**One-liner:** The main event: Unsloth + DoRA + 15K SQL on A100 → postgres-sqlcoder-7b-v1 in ~20 minutes.
