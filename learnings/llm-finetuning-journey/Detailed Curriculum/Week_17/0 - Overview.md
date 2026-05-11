# Week 17 Overview — Scaling Laws

This file is your map for Week 17. Read it first; everything else fits inside it.

## The story this week

Scaling laws describe how model performance (measured as loss on a held-out set) changes predictably as you increase model parameters (N), training tokens (D), or compute (C). If you can predict loss from N and D, you can plan experiments without running them.

## What you need to do

- [ ] Papers downloaded and skimmed: Kaplan 2020, Chinchilla 2022
- [ ] Calculator or Python notebook ready (no GPU needed this week)
- [ ] GitHub repo with a `week-17-scaling-laws/` directory

Concretely, by the end of the week you should be able to:

- Explain the Kaplan (2020) scaling law findings and their limitations
- Apply the Chinchilla (2022) compute-optimal formula to choose model size and token count for a given compute budget
- Distinguish between compute-optimal training and inference-optimal training
- Calculate the approximate FLOP cost of a training run given model parameters and tokens
- Identify why over-parameterized, under-trained models were common before Chinchilla

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

- 2h: Read Kaplan 2020 (Sections 1–4 are enough; skim the rest)
- 2h: Read Chinchilla 2022 (focus on Sections 2–4 and Table A3)
- 1h: Watch Yannic Kilcher's Chinchilla walkthrough
- 1.5h: Write your deliverable — the compute budget writeup applying Chinchilla to your $50 Phase 6 budget
- 0.5h: Sanity-check your math against published model cards (GPT-3, Llama-1, Chinchilla)

## Why this week matters

**One-liner:** Chinchilla says scale tokens and parameters equally; inference needs justify over-training smaller models.
