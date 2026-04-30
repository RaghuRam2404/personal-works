# Week 69 Assignment — Technical Report: Evaluation and Ablations

## Setup Checklist

- [ ] Evaluation results from Weeks 61–62 consolidated into `results/all_results.csv`
- [ ] Intermediate checkpoints available (CPT-only, SFT-only, DPO-only) for ablation runs; if not, note which ablations will be estimated
- [ ] `report/report_draft_v1.md` from Week 68 ready
- [ ] A sample of 30–40 failed examples from your custom 200-example benchmark identified

## Task 1: Consolidate Results Table

**Goal:** A single, authoritative results table covering all models and benchmarks.

**Requirements:**
- [ ] Create `results/main_results_table.md` with all models as rows, all benchmarks as columns
- [ ] Include model version strings and evaluation dates for all closed-source models
- [ ] Mark with † any result taken from published sources (not measured by you)
- [ ] Mark with — any combination you did not run
- [ ] Bold the best result per column
- [ ] Include a footnote explaining your metric (exact-match or execution accuracy) for each benchmark

**Deliverable:** `results/main_results_table.md`

## Task 2: Write the Evaluation Section

**Goal:** A 600–900 word evaluation section covering setup and main results.

**Requirements:**
- [ ] Section 5.1.1: For each benchmark, provide: name, size, schema domain, metric used
- [ ] Section 5.1.2: Define exact-match and execution accuracy mathematically; state which one you used and why
- [ ] Section 5.1.3: List all baselines with checkpoint/version strings
- [ ] Section 5.1.4: Report inference settings (temperature=0.1, max_tokens=512, no sampling tricks)
- [ ] Section 5.2: Present the results table with 3–4 paragraphs of analysis — what patterns you see, where your model wins, where it loses
- [ ] Analysis must include: comparison to base model, comparison to next-best open-weight, comparison to GPT-4o

**Deliverable:** `report/evaluation_section.md`

## Task 3: Write the Ablation Study

**Goal:** A 300–500 word ablation section with a table showing each stage's contribution.

**Requirements:**
- [ ] Ablation table: rows = configurations (base, +CPT, +SFT, +DPO, +GRPO), columns = Custom-200 + BIRD-SQL dev
- [ ] If you do not have a checkpoint for an ablation stage, explicitly mark it as "checkpoint not saved — estimated from nearest available" and describe your estimation method
- [ ] Write 2–3 paragraphs analyzing what each stage contributed and which stage provided the most benefit
- [ ] Include at least one additional ablation dimension (e.g., SFT data ablation OR DPO β OR GRPO K)

**Deliverable:** `report/ablation_section.md`

## Task 4: Write the Failure Mode Analysis

**Goal:** A categorized analysis of your model's errors that directly motivates the Limitations section.

**Requirements:**
- [ ] Manually inspect 30–40 failed examples from your Custom-200 benchmark
- [ ] Categorize failures into 4–6 error types; quantify each (N examples, % of failures)
- [ ] For each error type, show one representative example (schema snippet + question + wrong output + correct output)
- [ ] Identify which error types are addressed by existing work and which are novel to the TimescaleDB domain
- [ ] Save as `report/failure_analysis.md`

**Deliverable:** `report/failure_analysis.md` (integrated into Section 5.3 of the report)

## Stretch Goals

- Run execution accuracy evaluation on your Custom-200 benchmark against a local PostgreSQL instance; compare EM vs EX accuracy to quantify how often semantically equivalent SQL differs in string form
- Create a confusion matrix of SQL clause types in errors (which clauses are most often wrong)
- Write a short appendix section listing all 200 benchmark examples with their model outputs and pass/fail labels
