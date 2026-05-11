# Week 77 Overview — Curriculum — Bilingual NL→SQL: English + Tamil

This file is your map for Week 77. Read it first; everything else fits inside it.

## The story this week

Tamil is one of the world's oldest classical languages with over 70 million native speakers, predominantly in Tamil Nadu (India) and Sri Lanka. As you build tools for Indian enterprises and government systems, the ability to query databases in Tamil is commercially and socially significant. The standard NL→SQL pipeline assumes English input; extending it to Tamil creates three distinct challenges: tokenizer coverage, training data availability, and cross-lingual generalization.

## What you need to do

- [ ] Python environment with `transformers>=4.40`, `trl>=0.8`, `peft>=0.10`, `datasets`
- [ ] Access to a translation tool: IndicTrans2 (preferred, open-source), Google Translate API, or GPT-4o API for Tamil translation
- [ ] Your existing SFT training set (from Week 55 or later) accessible locally
- [ ] Your fine-tuned model checkpoint (`postgres-sqlcoder-7b-final`) available
- [ ] W&B project `week-77-bilingual-sql` created
- [ ] Optional: A Tamil speaker to spot-check 20–30 translations (reach out to a friend or use a Tamil online community forum for quick checks)

Concretely, by the end of the week you should be able to:

- Assess the tokenizer coverage of your base model for Tamil script and identify the practical consequences of poor coverage.
- Build a parallel bilingual NL→SQL dataset with Tamil question variants for your existing training examples.
- Fine-tune your model on bilingual data with a balanced mixing strategy that preserves English performance while adding Tamil.
- Evaluate your model across both languages on Custom-200 and diagnose per-language failure patterns.
- Articulate the limits of the current approach and what a production-grade bilingual system would require.

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

- 1 hour: Tokenizer coverage analysis; measure token inflation for Tamil vs English; document findings.
- 1.5 hours: Build Tamil training set (translate 300 English questions, spot-check 30 manually).
- 1.5 hours: Fine-tune on bilingual mix (90/10 English/Tamil, 1000 steps); log to W&B `week-77-bilingual-sql`.
- 1.5 hours: Evaluate on Custom-200 in both English and Tamil; compute per-language EM; diagnose failure patterns.
- 0.5 hours: Write results memo documenting performance gap and production requirements.

## Why this week matters

**This week in 15 words:** Tamil NL→SQL is feasible as a prototype but requires honest acknowledgment of tokenizer and data constraints.
