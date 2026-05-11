# Week 33 Overview — QLoRA: Your First 7B Fine-Tune

This file is your map for Week 33. Read it first; everything else fits inside it.

## The story this week

QLoRA (Dettmers et al., 2023) is elegantly simple in implementation despite its theoretical sophistication:

## What you need to do

- [ ] Colab Pro with A100 runtime selected (required for 7B training)
- [ ] `pip install trl peft transformers bitsandbytes datasets accelerate wandb`
- [ ] 5K training examples formatted (reuse Week 29/31 dataset)
- [ ] 100-example held-out test set from Week 32 (`held_out_test.json`)
- [ ] W&B project `week-33-qlora-7b` created
- [ ] HuggingFace write token set

Concretely, by the end of the week you should be able to:

- Combine bitsandbytes 4-bit NF4 quantization with peft LoRA into a single QLoRA training pipeline
- Fine-tune `Qwen2.5-Coder-7B` on your domain dataset in under 30 minutes on a Colab Pro A100
- Explain why QLoRA uses bfloat16 for LoRA adapter compute despite an INT4 base model
- Evaluate a fine-tuned 7B model on your held-out 100-example SQL test set
- Push a 7B adapter to HuggingFace Hub

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
| Read QLoRA paper (fully) | 2h |
| Set up Colab Pro A100 (request if needed) | 30m |
| Write and debug QLoRA training script | 2h |
| Run training (estimate 20–40 min on A100 with 5K examples) | 1h |
| Evaluate on held-out test set, push to HuggingFace | 1h |
| Commit and document results | 30m |

## Why this week matters

**One-liner:** QLoRA = 4-bit frozen base + BF16 LoRA adapters. Train 7B on 24GB. Gradients never touch the NF4 weights.
