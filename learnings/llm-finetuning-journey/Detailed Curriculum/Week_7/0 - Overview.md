# Week 7 Overview — HuggingFace Ecosystem Onboarding

This file is your map for Week 7. Read it first; everything else fits inside it.

## The story this week

HuggingFace consists of four main libraries you will use throughout this course:

## What you need to do

- [ ] HuggingFace account created and `huggingface-cli login` works. Verify: `huggingface-cli whoami` prints your username.
- [ ] `pip install transformers datasets huggingface_hub` (or upgrade to latest).
- [ ] HF_TOKEN environment variable set (or logged in via CLI). Check with `python -c "from huggingface_hub import whoami; print(whoami())"`.
- [ ] Colab Free available for the heavier inference tasks. Mac is fine for tokenization tasks.

Concretely, by the end of the week you should be able to:

- Load any model from the HuggingFace Hub using `AutoModel` and `AutoTokenizer`, run inference, and inspect the output logits.
- Use the `datasets` library to load, filter, map, and split standard datasets — including Spider.
- Understand and use the `attention_mask` and `labels` fields required for language model training with HuggingFace `Trainer`.
- Push a tokenized dataset to your HuggingFace account as a private dataset artifact.
- Identify which components of the HuggingFace stack you will use in each future phase and how they relate to raw PyTorch.
- Explain the `AutoClass` pattern, `PreTrainedModel`, and `PreTrainedTokenizerFast` class hierarchy.

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
| Read HuggingFace LLM Course Chapters 1–3, do all exercises | 2 h |
| Load distilgpt2, run inference, inspect logits | 45 min |
| Load Spider from datasets hub, explore structure | 30 min |
| Tokenize Spider with Qwen2.5-Coder tokenizer, push to Hub | 1.5 h |
| Push tokenized dataset to your HuggingFace account | 30 min |
| Explore `model.generate()` parameters (temperature, top-k, top-p) | 45 min |
| Journal + commit | 30 min |

## Why this week matters

**This week in 15 words:** HuggingFace wraps PyTorch; know its API but never forget what's underneath.
