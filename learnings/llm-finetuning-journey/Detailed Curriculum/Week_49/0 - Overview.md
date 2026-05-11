# Week 49 Overview — KTO, ORPO, and the Alignment Zoo

This file is your map for Week 49. Read it first; everything else fits inside it.

## The story this week

Week 49 continues the curriculum's thread; the Curriculum file explains the conceptual setup in detail.

## What you need to do

- [ ] Papers open: KTO (2402.01306), ORPO (2403.07691), SimPO (2405.14734)
- [ ] Phase 5 notes from Weeks 41–48 available for reference
- [ ] No GPU required this week

Concretely, by the end of the week you should be able to:

- Describe KTO, ORPO, and SimPO at a conceptual level and identify what problem each solves
- Produce a comparison table of all alignment methods covered in Phase 5 (PPO, DPO, GRPO, KTO, ORPO, SimPO)
- Articulate for each method: the data requirement, the loss formulation at a high level, and the ideal use case
- Explain why you chose GRPO for your SQL domain (and when you would choose each alternative)
- Use this week to consolidate Phase 5 knowledge before the Gate (Week 52)

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

- 1.5 hours: Skim KTO paper (abstract, Section 2, Table 1 in the paper).
- 1 hour: Skim ORPO paper (abstract, Section 3 on the loss, experiments).
- 1 hour: Skim SimPO paper (abstract, Section 3, compare to DPO results table).
- 1 hour: Build the comparison table (Assignment Task 1).
- 1 hour: Write the "For my SQL domain" analysis (Assignment Task 2).
- 1–2 hours: TRL docs browsing — find KTOTrainer, ORPOTrainer, SimPOTrainer. Note the config differences.

## Why this week matters

**One-liner:** PPO/GRPO = online; DPO/KTO/ORPO/SimPO = offline. Use verifiable rewards → GRPO; human prefs → DPO; unpaired labels → KTO.
