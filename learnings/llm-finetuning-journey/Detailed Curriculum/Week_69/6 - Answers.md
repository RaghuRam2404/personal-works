# Week 69 Answers

## Q1

**Answer: B**

**Why correct:** Two SQL queries can be semantically equivalent (produce the same result set) while being syntactically different — they may use different column aliases, different join orderings, different formatting, or equivalent but distinct aggregation expressions. Exact-match penalizes all of these as wrong, even when the query would produce the correct results. Execution accuracy measures what matters: does the query return the right rows? EX is more meaningful but harder to implement because it requires a live database with test data. Most papers use EX when possible and clearly state which metric was used.

**Why others are wrong:**
- A: EM requires only string comparison; EX requires a database — EM is the cheaper metric.
- C: EM is simpler, not harder.
- D: EX works for TimescaleDB; it requires a TimescaleDB test database, which you set up in your evaluation harness.

---

## Q2

**Answer: B**

**Why correct:** A binary classification test on N=200 examples has a standard error of approximately sqrt(p(1-p)/N). At p=0.83, this is sqrt(0.83×0.17/200) ≈ 0.0266, giving a 95% CI of ±2×0.027 ≈ ±5.3 pp (or ±3.1 pp for one-sided). A 0.5 pp improvement falls well within this confidence interval, meaning it cannot be distinguished from noise at the 95% level. The correct approach is to (a) acknowledge this, (b) report the CI, and (c) note the improvement is consistent across multiple benchmark runs if you ran them. Do not remove the ablation — it is still informative data even if not statistically conclusive.

**Why others are wrong:**
- A: Removing a valid ablation because of statistical noise would be wrong; the data is still evidence even if uncertain.
- C: Running multiple inference passes on a deterministic model (temperature=0.1) will give nearly identical results — you cannot reduce variance this way.
- D: Switching benchmarks for only the ablation breaks consistency with the main results table.

---

## Q3

**Answer: C**

**Why correct:** The failure mode analysis showed that 35% of errors involve missing `time_bucket` — the model is not generating this TimescaleDB-specific function often enough. The direct fix is supervised: add more `time_bucket` examples to the training data and re-run SFT. This targets the exact pattern the model is failing on. DPO or GRPO modifications do not address the root cause (lack of training signal for `time_bucket`).

**Why others are wrong:**
- A: GRPO vs PPO does not affect what SQL syntax patterns the model has learned.
- B: Changing β affects the DPO training dynamics, not what SQL functions the model knows.
- D: A larger model may learn `time_bucket` better from the same data, but it does not directly address the training data gap.

---

## Q4

**Answer: A** (and then B if A is not feasible in your timeline)

**Why correct:** The most rigorous answer is A: run SQLCoder-7B on Custom-200 yourself, because benchmarks are not directly comparable across different test sets. Your Custom-200 is specifically designed for TimescaleDB; it may be easier or harder than Spider in ways that make cross-benchmark comparison misleading. However, if running SQLCoder-7B is outside your week's scope, B is the correct journalistic approach: report what you can verify (their published Spider result with †), leave blank what you did not measure, and explicitly note in the paper that Custom-200 comparisons require running models on your test set.

**Why others are wrong:**
- C: Interpolating performance from a different benchmark is not a valid scientific practice.
- D: Asking the team is reasonable but not a substitute for running the evaluation yourself.

---

## Q5

**Model answer:** Exact-match normalization refers to a sequence of transformations applied to both generated and reference SQL strings before comparison, so that syntactically different but logically equivalent SQL does not count as wrong. Three universally applied normalizations: (1) Lowercase the entire SQL string — `SELECT` and `select` are the same keyword. (2) Strip leading/trailing whitespace and collapse multiple spaces to a single space. (3) Remove semicolons at the end of the statement — some datasets include them, some do not. Additional normalizations used in Spider/BIRD evaluations include: removing column aliases if they are not part of the answer, normalizing `!=` to `<>`, and sorting `SELECT` columns alphabetically when the order does not matter. Document exactly which normalizations you apply.

---

## Q6

**Model answer:** To isolate whether CPT content vs compute time drives the 6.3 pp gain, run a control experiment: continue pretraining on the same number of tokens (102M) but from a random-text corpus (e.g., Wikipedia or a random Common Crawl sample) instead of PostgreSQL-domain text. If the control gives similar accuracy improvement to PostgreSQL CPT, then the gain is from compute (more gradient updates), not from domain content. If the control gives much smaller gain, the domain content is what matters. This is called a "content vs compute ablation" and is standard practice in CPT papers. In your case, you would expect PostgreSQL CPT to dominate because the base model already had extensive code pretraining — additional generic text provides little new signal, whereas domain SQL text is underrepresented in the original corpus.

---

## Q7

**Model answer:** The most important caveat is: "on our Custom-200 TimescaleDB benchmark, using our specific prompt format and evaluation protocol, measured in December 2024." This claim does not generalize to: (1) other SQL benchmarks (BIRD, Spider, Defog) where GPT-4o still leads or is competitive; (2) GPT-4o's current version, which may have been updated since evaluation; (3) general SQL tasks outside the TimescaleDB domain; (4) any inference settings other than temperature=0.1 and your exact prompt template. Benchmark-specific superiority is a narrow and honest claim; "beats GPT-4o at SQL" without qualification overstates it significantly and invites justified criticism.

---

## Q8 — Deep Scenario

**Model answer:** Two possible sources of the 4.2 pp gap: First, their pipeline may be genuinely better — they may have used a larger or higher-quality training dataset, a different training stage order, stronger rewards in GRPO, or longer training. Second, their evaluation may differ from yours — they may have evaluated with a different prompt format, temperature, or normalization that happens to benefit their model's output style, or they may have used execution accuracy while you used exact-match (inflating their number relative to yours).

One-week experiment: request (or reproduce) their evaluation script and run it on your model using their exact inference settings. If your model also achieves 87%+ under their evaluation conditions, the gap was methodological, not a real quality difference. If your model still scores 83% under their settings, the gap is genuine.

This result does not invalidate your technical report. Your contribution includes the benchmark, the dataset, the training recipe, and the quantized deployable variants — none of which are negated by a subsequent model achieving higher accuracy. Science is incremental; a model that outperforms yours is evidence that your work provided a useful foundation, not that it was wrong.
