# Week 68 Overview — Technical Report Week 2: Training Pipeline and Architecture

This file is your map for Week 68. Read it first; everything else fits inside it.

## The story this week

The training section answers: what did you do to the base model, in what order, with what hyperparameters, and how long did it take? Readers need enough detail to reproduce your pipeline. The section should not re-explain how transformers work — assume the reader knows. Instead, describe your specific choices and why you made them.

## What you need to do

- [ ] W&B project open for all four training runs (Weeks 57–60)
- [ ] `report/introduction.md` and `report/dataset_section.md` from Week 67 ready
- [ ] Exact hyperparameter configs available (W&B config tab, or your training scripts)
- [ ] GPU-hour logs or RunPod billing dashboard for compute budget

Concretely, by the end of the week you should be able to:

- Write a training pipeline section that documents all four training stages with reproducibility-grade detail
- Describe architecture decisions clearly without redundant base-model description
- Present hyperparameter tables in the format expected by ML publication venues
- Write a compute budget section that translates your training runs into GPU-hours
- Integrate the training section into your growing report document

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

- 0.5h: Review W&B config logs from Weeks 57–60 and extract exact hyperparameters
- 1.0h: Write base model + CPT subsection
- 1.5h: Write SFT + DPO subsections
- 1.0h: Write GRPO subsection
- 1.0h: Build hyperparameter table
- 0.5h: Write compute budget section
- 0.5h: Integrate into main `report.md` document and review for consistency

## Why this week matters

Every number in the training section must trace to a W&B config tab, not to memory.
