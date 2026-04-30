# Course Generation Style Guide

This is the shared style guide for all 78 weeks of the LLM Fine-Tuning Course.
ALL subagents MUST follow this exactly to keep output consistent.

## User context (use second person, "you")
- Raghuram, software engineer in India, 6–8 hrs/week
- Already knows: linear algebra, basics of DL, basic NN in plain Python, basics of attention (theory only), basics of transfer learning (theory only)
- Knows nothing of: PyTorch hands-on, RL, transformers in code, fine-tuning in code, quantization
- Domain target: PostgreSQL/TimescaleDB Text-to-SQL
- Compute: Mac (Apple Silicon, MPS), Colab Free → Pro, RunPod for late phases
- Framework: PyTorch ONLY

## Per-week files (8 files per Week_N folder)

### 1. Curriculum.md
Detailed concepts to learn that week. Should expand on the source curriculum (the master file at `/home/user/workspace/llm_finetuning_curriculum_18months.md`), not just copy it. Include:
- **Learning objectives** (3–6 bullet points, "By end of this week, you will be able to...")
- **Concepts** with sub-headings — explain each concept in 1–3 paragraphs with intuition + math when relevant
- **Connections** — how this week builds on prior weeks and what later weeks will need
- **Common misconceptions / pitfalls** — what trips beginners up
- **Time allocation guidance** — how to spend the 6–8 hrs (e.g., 2h reading, 1h video, 4h coding)
- Length: 600–1500 words

### 2. Assignment.md
Things the user must try themselves. Format:
- **Setup checklist** (env, datasets, models needed)
- **Tasks** numbered 1, 2, 3... with explicit requirements (input/output specs, what to log, what to commit)
- Each task has: **Goal**, **Requirements** (bullet list — concrete, testable), **Deliverable** (file path or commit message), optionally **Hints** section if user must reason about something
- **Stretch goals** (optional, for users with extra time)
- Use real numbers: "achieve val loss < 2.5", "run for at least 1000 steps", "log to W&B project name `week-N-<topic>`"
- Length: 400–1000 words

### 3. AssignmentSolutions.md
Essentials only. NOT full implementations. Format:
- For each task: **Key snippets** (10–30 lines max showing the trickiest part), **Expected output** (loss range, accuracy, sample text), **Common gotchas** (3–5 bullets of debugging tips)
- Add a **"How to verify you did it right"** section
- Length: 300–700 words

### 4. Quiz.md
Calibrated to phase difficulty:
- **Phase 1–2 (Weeks 1–16):** Junior ML interview level — 5–7 MCQ + 2–3 short-answer
- **Phase 3–4 (Weeks 17–40):** Mid-level interview — 4–6 MCQ + 3–4 short-answer + 1 scenario
- **Phase 5–6 (Weeks 41–78):** Senior research engineer — 3–5 MCQ + 3–5 short-answer + 1–2 deep scenarios
- Questions must be **real-world / scenario-based**, not trivia. Example: "You're fine-tuning Qwen-7B and loss diverges at step 200. List 4 hypotheses ranked by likelihood." NOT "What does LR stand for?"
- Number every question Q1, Q2, Q3...
- For MCQ: provide 4 options labeled A/B/C/D
- Length: 400–800 words

### 5. Answers.md
Answers with explanations. For each Q:
- **Answer**: the correct answer
- **Why**: 2–4 sentence explanation of reasoning
- **Why others are wrong** (for MCQ): brief note on each distractor
- For scenario questions: a model answer paragraph
- Length: 500–1000 words

### 6. TakeAway.md
The cheatsheet. Dense, scannable, no fluff. Format:
- **One-liner** at top: what this week was about in 15 words
- **Key formulas** (if any) in code blocks
- **Key code patterns** (5–15 line snippets the user will reuse)
- **Decision rules**: "If X, then Y" type guidance
- **Numbers to remember** (e.g., "1/sqrt(d_k) scaling factor", "AdamW default LR for fine-tuning: 2e-4 with LoRA")
- **Red flags during training** (signs something's wrong)
- Length: 200–500 words. SHORT and DENSE.

### 7. Glossary.md
Terms introduced THIS WEEK only (don't repeat prior weeks). Format:
```
**Term**: 1-line definition (max 20 words)
```
6–15 terms per week typically. More in dense weeks.

### 8. Resources.md
Dedicated link dump. Format with sections:
- **Papers** (with arXiv link + 1-line description)
- **Videos** (with YouTube link, channel, duration)
- **Blog posts / Articles**
- **GitHub repos**
- **Documentation**
- **Optional / Bonus**

Group by section. Use markdown links: `[Title](url)`. Verify links exist as best you can — these were verified as of April 2026.

## Universal style rules
- **Voice**: Second person ("you"), warm but rigorous. Like a strict but caring senior engineer mentor.
- **No fluff**: Every sentence earns its place. No "in this exciting week, we will explore..."
- **Concrete > abstract**: Real numbers, real model names, real commands.
- **PyTorch only** for all code. No TF, no JAX.
- **Domain hooks**: When natural, tie examples back to PostgreSQL/SQL/text-to-SQL. Don't force it.
- **Markdown formatting**: Use `##` for sections, `###` for subsections, code fences with language tags (```python), tables when comparing things.
- **NO emojis.**
- **NO italic markdown** (don't use `*text*`).
- **Citations inline**: When linking external resources, use natural anchor text: `[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)` not `[here](url)`.

## What "calibrated to phase" means concretely

| Phase | Weeks | Quiz tone |
|---|---|---|
| 1 | 1–8 | Junior ML eng. "Why does ReLU help vs sigmoid?" "Diagnose this loss curve." |
| 2 | 9–16 | Mid-junior. "Implement scaled dot-product attention. Why divide by sqrt(d_k)?" |
| 3 | 17–27 | Mid. "Given $50 compute budget, pick model size and tokens via Chinchilla." |
| 4 | 28–40 | Mid-senior. "Choose LoRA rank for 7B model on 10k examples; justify." |
| 5 | 41–52 | Senior applied. "Design a reward function for SQL with executable verification + reasoning bonus." |
| 6 | 53–78 | Senior research. "Your 7B model beats GPT-4 on Spider but loses on TimescaleDB. Diagnose and propose 3 interventions." |

## File naming
Exact filenames in each Week_N folder:
- `Curriculum.md`
- `Assignment.md`
- `AssignmentSolutions.md`
- `Quiz.md`
- `Answers.md`
- `TakeAway.md`
- `Glossary.md`
- `Resources.md`

## Output discipline
- Write each file with the `write` tool to `/home/user/workspace/llm_course/Week_N/<filename>.md`
- Do NOT batch-write across weeks — one week at a time, commit each before moving on
- After each week, briefly note in your scratchpad what was covered so you don't repeat across weeks

## Pacing per week (generation guide for subagents)
Aim for ~3000–5000 words total per week across the 8 files combined. Some weeks (heavy coding weeks) may be larger; some (reading weeks) may be smaller.
