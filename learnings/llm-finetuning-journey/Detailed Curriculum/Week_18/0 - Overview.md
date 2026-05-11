# Week 18 Overview — Pretraining Data: Sources, Filtering, and Deduplication

This file is your map for Week 18. Read it first; everything else fits inside it.

## The story this week

[Common Crawl](https://commoncrawl.org/) is a non-profit that crawls the public web monthly and releases the data for free. As of 2024, the archive is over 250 petabytes. It is the primary source for virtually every modern pretraining dataset.

## What you need to do

- [ ] `pip install datasets datasketch fasttext-wheel langdetect trafilatura`
- [ ] HuggingFace account and `huggingface_hub` configured (`huggingface-cli login`)
- [ ] Colab Free notebook OR local environment with ~4GB RAM available
- [ ] GitHub repo with `week-18-pretraining-data/` directory

Concretely, by the end of the week you should be able to:

- Describe the lineage of major pretraining datasets: Common Crawl → C4 → The Pile → RefinedWeb → FineWeb
- Explain the key quality filtering steps applied to raw web crawl data
- Implement MinHash-based near-duplicate detection on a text corpus
- Run basic data statistics (length distribution, language identification) on FineWeb-Edu
- Articulate why data quality is often the decisive variable in modern LLM training

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

- 1h: Read The Pile paper (Sections 1–3, skim the rest)
- 1h: Read RefinedWeb paper (Sections 1–4)
- 1h: Read FineWeb dataset card + technical blog post
- 1.5h: Download FineWeb-Edu shard, run basic statistics
- 2h: Implement MinHash deduplication using `datasketch`
- 0.5h: Commit and write notes in `journal.md`

## Why this week matters

**One-liner:** High-quality filtered Common Crawl beats diverse dirty data; MinHash catches near-duplicates that destroy training.
