# Week 6 Overview — Tokenization Deep Dive

This file is your map for Week 6. Read it first; everything else fits inside it.

## The story this week

Neural networks operate on fixed-size vectors. Text must be mapped to integers (token IDs) that index into an embedding table. The question is: what unit should a "token" represent?

## What you need to do

- [ ] Install `tiktoken`: `pip install tiktoken`.
- [ ] Install HuggingFace `tokenizers`: `pip install tokenizers`.
- [ ] Spider SQL corpus available: `week_04/sql_queries.txt` (raw SQL query strings). If not, re-extract from Spider.
- [ ] Karpathy minbpe repo cloned for reference (do NOT copy-paste): `git clone https://github.com/karpathy/minbpe`.
- [ ] W&B not needed this week.

Concretely, by the end of the week you should be able to:

- Explain why character-level and word-level tokenization are suboptimal for LLMs, and why subword tokenization is the current standard.
- Implement byte-pair encoding (BPE) from scratch, including the merge loop and vocabulary construction.
- Reproduce GPT-4's tokenization regex patterns from Karpathy's `minbpe` and explain each component.
- Train a BPE tokenizer on a SQL corpus and compare its vocabulary to GPT-4's tokenization of the same SQL.
- Explain four concrete ways that tokenization choices affect model behavior for code and SQL generation.
- Use HuggingFace `tokenizers` to load, inspect, and extend a pre-trained tokenizer.

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
| Watch Karpathy's GPT Tokenizer video — code along (2h13m) | 2.5 h |
| Read HuggingFace Tokenizers tutorial Chapters 1–3 of the LLM course section on tokenization | 1 h |
| Implement BPE from scratch (training loop) | 1.5 h |
| Train BPE on SQL corpus; compare to GPT-4 | 1 h |
| Journal + commit | 30 min |

## Why this week matters

**This week in 15 words:** BPE merges byte pairs iteratively; tokenization choices affect SQL generation quality in production.
