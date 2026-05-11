# Week 55 Assignment — Filter Raw Data Aggressively

## Setup Checklist

- [ ] Raw dataset from Week 54 (`v3_raw.jsonl`) accessible locally
- [ ] PostgreSQL with all test schemas loaded
- [ ] `openai` client configured; API key set
- [ ] 50 hand-annotated examples ready (your gold calibration set)
- [ ] W&B project `week-55-filtering` created

---

## Task 1 — Build the Calibration Set

**Goal:** Ground your LLM judge in human judgment before scaling to 30K examples.

**Requirements:**
- Randomly sample 100 examples from `v3_raw.jsonl` (execution-passing only)
- Manually label each as: `keep` (would use for training) or `reject` (would not use)
- Record your reason for each `reject`: wrong SQL / ambiguous question / too trivial / irrelevant
- Save as `calibration_set.jsonl` with fields: `id`, `question`, `sql`, `schema`, `human_label`, `human_reason`

**Deliverable:** `calibration_set.jsonl` committed with exactly 100 labeled examples.

---

## Task 2 — LLM Judge Prompt Design

**Goal:** Write a judge prompt that agrees with your human labels at rate ≥ 80%.

**Requirements:**
- Write `judge_prompt.txt` using this structure:
  - System: judge role, domain context, evaluation mandate
  - Schema DDL: `{schema_ddl}`
  - Question: `{question}`
  - SQL: `{sql}`
  - Rubric: explicit 1–5 definitions (see Curriculum)
  - 2 few-shot examples (one score-5, one score-2) with explanations
  - Output format: `{"score": <1-5>, "reason": "<one sentence>"}`
- Run the judge on all 100 calibration examples
- Compute: agreement rate (judge ≥ 4 ↔ human "keep") and Cohen's kappa
- If agreement < 80%: refine rubric and repeat
- Save final metrics in `judge_calibration_results.md`

**Hints:**
- Use `sklearn.metrics.cohen_kappa_score` for kappa
- The most common calibration failure: judge scores verbose-but-wrong SQL highly because it "looks professional." Add to rubric: "Correctness dominates style. A concise wrong SQL scores 1. A verbose correct SQL scores 4, not 5."

**Deliverable:** `judge_prompt.txt` + `judge_calibration_results.md` committed.

---

## Task 3 — Full Filtering Pipeline

**Goal:** Filter 30K raw examples to a high-quality subset using all signals.

**Requirements:**
Write `filter_v3.py` that applies filters in this order:
1. Skip if `execution_status != "pass"` (already done in Week 54, but re-validate)
2. Skip if `is_duplicate == True` (already done in Week 54)
3. Skip if result set is empty (`SELECT COUNT(*) = 0`)
4. Skip if complexity-difficulty mismatch (e.g., AST depth < 5 for "Hard" examples)
5. Run LLM judge; skip if score < 4 (or score < 3 for rare skills with < 200 examples)

For each rejected example, save to `v3_rejected.jsonl` with: original fields + `filter_stage` (which filter rejected it) + `judge_score` (if reached stage 5)

Log to W&B per-stage attrition numbers and per-skill counts after filtering.

**Deliverable:** `filter_v3.py` + `v3_filtered.jsonl` + `v3_rejected.jsonl` committed.

---

## Task 4 — Post-Filter Audit and Gap Remediation

**Goal:** Ensure no skill falls below the minimum threshold after filtering.

**Requirements:**
- Run your audit script from Week 53 on `v3_filtered.jsonl`
- Identify any skill with < 200 examples remaining
- For each under-represented skill: decide to either
  - Relax the judge threshold to score ≥ 3 for that skill (and add those examples back)
  - Hand-write 20–50 verified examples for that skill
- Document decisions in `post_filter_gap_analysis.md`

**Acceptance criteria:**
- Final `v3_filtered.jsonl`: at least 20,000 examples
- No skill with < 150 examples
- TimescaleDB-specific examples: at least 1,500
- Overall judge score distribution logged to W&B

**Deliverable:** `post_filter_gap_analysis.md` + updated `v3_filtered.jsonl` + push to HuggingFace as `<your-handle>/postgres-sql-v3`.

---

## Stretch Goals

- Implement a "self-consistency" filter: run the same question through your current v2 model and a simple symbolic SQL checker (e.g., `sqlglot` can validate some logic). If the v2 model gets it right, the example is "too easy" and may be deprioritized in training.
- Build a score histogram visualization and embed it in your data card — this shows reviewers of your technical report that your data quality is serious.
- For borderline examples (score = 3), apply a secondary judge (Claude 3.5 Sonnet) and keep only if both judges rate it 3+. This two-judge consensus reduces false positives.
