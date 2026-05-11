# Week 70 — Technical Report Week 4: Limitations, Polish, and Publish

## Learning Objectives

By the end of this week, you will be able to:

- Write a limitations section that is honest without being self-defeating
- Produce a reproducibility appendix that meets NeurIPS/EMNLP checklist standards
- Integrate all four sections into a final polished report document
- Submit a preprint to arXiv or release as a HuggingFace model card technical report
- Write a future work section that is specific and actionable, not vague

## Concepts

### Writing the Limitations Section

A limitations section is not a confession of failure — it is evidence that you understand your work's boundaries. Strong limitations sections identify specific gaps, explain their impact, and suggest where future work should address them. Weak limitations sections are vague ("our model may not generalize") or self-defeating ("our results may be wrong").

Structure for your limitations section:

Limitation 1: Single-domain generalization. Your model was fine-tuned on PostgreSQL/TimescaleDB data. It may underperform on MySQL, SQLite, or Snowflake syntax, which differ in date functions, LIMIT syntax, and window function support. You have not evaluated on these databases.

Limitation 2: Benchmark coverage. Your Custom-200 benchmark covers TimescaleDB time-series functions well but under-covers: lateral joins, recursive CTEs, full-text search (`tsvector`/`tsquery`), and PostGIS geographic functions. Accuracy on these patterns is unknown.

Limitation 3: Multi-turn and agentic SQL. Your model is trained and evaluated on single-turn question-to-SQL pairs. Real database interactions often involve iterative refinement ("the last query was wrong because..."). Multi-turn performance is not characterized (Week 76 will address this).

Limitation 4: Quantization accuracy gap. Your quantized variants (Q4_K_M GGUF, AWQ INT4) show 0.3–2.1 pp accuracy degradation on Custom-200 vs the BF16 model. For production systems requiring maximum accuracy, the quantization tradeoff must be evaluated per-deployment.

Limitation 5: Calibration. You did not study the model's calibration — whether its generation confidence (log-probability) correlates with correctness. Overconfident wrong SQL is more dangerous than low-confidence wrong SQL for automated execution.

### Writing Future Work

Future work is not a list of things you wish you had done — it is a research agenda. For each future direction, state: (a) what specifically you would do, (b) why you expect it to work, and (c) what you would measure.

"Multi-turn SQL refinement: Fine-tune on CoSQL (Yong et al. 2019) and your custom multi-turn dataset to teach the model to accept a previous SQL + error message as context and produce a corrected query. We expect 5–10 pp improvement on multi-turn benchmarks based on the analogous gain seen when CoSQL data was added to Spider baselines (published results)."

### Building the Appendix

The appendix is where reproducibility lives. It should include:

A. Full hyperparameter table (every non-default value for every stage)
B. Compute budget (per-stage GPU-hours, hardware, total cost)
C. Data release statement (dataset names, HuggingFace links, licenses)
D. Model release statement (all Hub repos, licenses, intended use)
E. Evaluation prompt template (the exact template used for all benchmarks)
F. Custom-200 benchmark description (how examples were created, who created them, when)
G. Reproducibility checklist (NeurIPS 2024 format)

The reproducibility checklist asks yes/no questions like: "Is all the training code released?" "Is the test set evaluation code released?" "Are all hyperparameters reported?" Walk through it line by line. If you answer No to anything, either fix it or explain why in the Appendix.

### Final Polish Pass

Before publishing, apply a systematic polish pass:

1. Consistency check: every number that appears in multiple places (abstract, intro, body, tables) must be identical. Run `grep "83.1" report.md` — it should appear consistently everywhere.

2. Citations: every claim that references another work has a citation. Every citation in the text appears in the reference list. No orphan references.

3. Table formatting: every table has a caption. Every table is referenced in the text by number ("Table 1 shows...").

4. Figure captions: every figure caption is self-contained — a reader who only reads the caption understands what the figure shows.

5. Tense consistency: methods in past tense, paper overview in present tense ("Section 4 describes..."), future work in future tense.

### Publishing to arXiv

arXiv is the standard venue for ML preprints. The process:

1. Create an account at arxiv.org (requires institutional affiliation or endorsement).
2. Prepare your submission: LaTeX source or PDF + figures. arXiv prefers LaTeX source.
3. Submit to cs.CL (Computation and Language) or cs.LG (Machine Learning) — both are appropriate.
4. Your paper gets a number (2412.XXXXX format) and appears within 1–2 business days.

Alternatively, publish as a HuggingFace model card technical report — attach the PDF to your model's repository under `report/`. This is acceptable and increasingly common for domain-specific fine-tuned models.

### Announcing the Release

Once the preprint is live:
- Tweet/post the arXiv link with: abstract, key number, HuggingFace model links
- Submit to Hugging Face's daily papers if you reach 83%+ on BIRD-SQL dev (their submission threshold for curation)
- Post to r/MachineLearning and r/LocalLLaMA

## Connections

This week completes the four-week report arc (Weeks 67–70). The report you publish this week is the capstone artifact for the entire 18-month course. Weeks 71–78 build on it: frontier reading weeks expand your knowledge beyond your own work, and iteration weeks (75–77) extend the model in new directions.

## Common Misconceptions / Pitfalls

The limitations section should not be the last paragraph of the paper written in five minutes. It is often the section that generates the most substantive reviewer feedback — a weak limitations section invites reviewers to identify your limitations for you, often less charitably than you would.

Do not publish the report before running the reproducibility checklist. A preprint with missing hyperparameters or broken code links reflects poorly and is very difficult to correct after publication.

Future work should not include things you plan to do in Weeks 75–77 — those are already in progress. Future work is directions beyond your current roadmap.

## Time Allocation (6–8 hours)

- 1.0h: Write limitations section (5 specific limitations, 50–80 words each)
- 0.5h: Write future work section (3 specific directions)
- 1.5h: Build appendix (hyperparameter table, data and model release statements, prompt template)
- 1.0h: Final consistency and polish pass
- 1.0h: LaTeX formatting or PDF generation
- 1.0h: arXiv or HuggingFace submission
- 0.5h: Announcement post
