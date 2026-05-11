# Week 36 Overview — DoRA, RSLoRA, LoftQ, and LoRA Variants

This file is your map for Week 36. Read it first; everything else fits inside it.

## The story this week

Standard LoRA (Week 30) is powerful but has known limitations:
- The rank determines capacity, but the scaling is naive (uniform `alpha/r` across all layers)
- At high ranks, LoRA may be numerically unstable or converge poorly
- When the base model is already quantized (QLoRA), the adapter may not optimally compensate for quantization artifacts

## What you need to do

- [ ] Colab Pro A100
- [ ] peft >= 0.9.0 (check: `pip show peft | grep Version` — DoRA requires peft 0.9+)
- [ ] Same 5K training / 200 eval / 100 held-out test split from previous weeks
- [ ] W&B project `week-36-lora-variants` created
- [ ] Unsloth installed (DoRA available via `use_dora=True` in `get_peft_model`)

Concretely, by the end of the week you should be able to:

- Explain the key differences between DoRA, RSLoRA, and LoftQ and the problem each solves
- Configure DoRA and RSLoRA via peft's `LoraConfig`
- Run a DoRA vs. LoRA comparison on your SQL dataset and interpret the results
- Decide when each variant is preferable over standard LoRA
- Understand the weight decomposition (magnitude + direction) in DoRA

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
| Read DoRA paper (arxiv 2402.09353) — sections 1–4 | 1.5h |
| Read RSLoRA paper (arxiv 2312.03732) — sections 1–3 | 1h |
| Skim LoftQ paper (arxiv 2310.08659) — sections 1–3 | 1h |
| Run DoRA vs. LoRA experiment | 2h |
| Write comparison report | 1h |
| Commit to GitHub | 30m |

## Why this week matters

**One-liner:** DoRA = magnitude+direction decomposition (usually +3–5% quality); RSLoRA = stable scaling for high ranks; LoftQ = compensate quantization error at init.
