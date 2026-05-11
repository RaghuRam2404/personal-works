# Week 62 Answers

## Q1 — Answer: B

**Why:** The TimescaleDB subset result is the most important for your project's stated goal — building a domain specialist that beats generalist models on PostgreSQL/TimescaleDB. A 5-point advantage (70% vs 65%) on the 50-example TimescaleDB subset is directly supportive of your thesis. The overall gap (76% vs 83%) should be reported honestly as a known limitation: generalist models perform better on general SQL due to broader training coverage. Both results together tell a complete, honest story.

**Why C is partially wrong:** 50 examples gives a CI of roughly ±14pp — the gap of 5pp is within noise. This should be disclosed. But the direction of the result (you win on your target domain) is meaningful.

---

## Q2 — Answer: B

**Why:** Both models are 7B-parameter SQL specialists, so architecture is similar. The meaningful differentiator is training data: SQLCoder was trained on a broad SQL corpus (Spider, BIRD, WikiSQL, Defog's enterprise data) while your model specialized on PostgreSQL/TimescaleDB specifically. This means: SQLCoder should generalize better across SQL dialects; your model should outperform on the specific PostgreSQL/TimescaleDB tasks. This comparison tests the "domain depth vs. breadth" trade-off directly.

---

## Q3 — Answer: A (with nuance)

**Why:** GPT-4o at temperature=0 is highly reproducible in practice but not guaranteed to be bit-for-bit identical across API calls due to floating-point ordering differences in parallel computation. For practical purposes, caching is valid: for the same input, GPT-4o will produce functionally identical outputs. However, for scientific rigor, running 3 trials on 10 random examples to verify reproducibility is worthwhile. If outputs are identical across trials, report this and justify the cache.

---

## Q4 — Cost Calculation

At equal accuracy (hypothetical):
- Your model: $0 per correct query (local inference, hardware already paid)
- GPT-4o: $0.03/0.83 = $0.036 per correct query

10,000 correct queries/month:
- Your model: $0 (ignoring electricity ~$2/month)
- GPT-4o: 10,000 × $0.036 = $360/month

Break-even: set your model's cost/correct = GPT-4o's cost/correct.
Your model accuracy: 76% = 0.76; GPT-4o accuracy: 83% = 0.83.
If GPT-4o price = P per query: cost/correct for GPT-4o = P/0.83
Cost/correct for your model = $0 (local)
Break-even at P = $0 — any non-zero cost makes GPT-4o more expensive per correct query than a free local model, regardless of accuracy difference.

At 10,000 queries/month where 76% succeed: you get 7,600 correct results for ~$0. GPT-4o gets 8,300 correct results for $300. The extra 700 correct answers cost $300 — $0.43 each. Whether that marginal cost is justified depends on the business value of each correct SQL query.

---

## Q5 — Model Answer

McNemar's test framework: count discordant pairs.
- Your model correct, GPT-4o wrong: 18 (b)
- GPT-4o correct, your model wrong: 9 (c)
- McNemar's test statistic: (|b - c| - 1)² / (b + c) = (|18 - 9| - 1)² / (18 + 9) = 64/27 ≈ 2.37

Critical value for chi-squared with 1 degree of freedom at α=0.05: 3.84.

2.37 < 3.84: NOT statistically significant at p < 0.05.

Conclusion: your model's advantage on the TimescaleDB subset (36% wins vs 18% losses) is in the right direction but not statistically significant with 50 examples. You need ~100 TimescaleDB examples to achieve 80% power to detect a 15pp win rate difference. Disclose this in your technical report: "The TimescaleDB-subset advantage is directionally consistent with our domain-specialization hypothesis but does not reach statistical significance at n=50."

---

## Q6 — Model Answer

If 30–40% of failures for both models are Type B (correct schema, wrong logic), those questions have an inherent difficulty that is not about schema knowledge — it is about SQL reasoning. These questions likely require multi-step inference: "which rows satisfy condition A AND their related rows in table B satisfy condition B, where the relationship is indirect." Both a 7B specialized model and a frontier generalist model struggle with this class of question.

This tells you: (a) the accuracy ceiling for these questions is not a data problem — more domain training data won't fix it without also improving reasoning capability; (b) the relevant intervention is reasoning-enhancing training (chain-of-thought SFT, GRPO with step-by-step reward) rather than more domain examples; (c) in your technical report, you can attribute a portion of the performance gap to "inherent question difficulty" rather than "model inadequacy."

---

## Q7 — Technical Pitch

Your 76% local model vs. GPT-4o's 83% is a 7-point gap in raw accuracy, but this framing misses the operational reality. At a typical PostgreSQL analytics platform serving 50,000 queries per month, GPT-4o's API costs exceed $1,500/month — a $18,000/year infrastructure cost that compounds as the platform scales. Our 7B model, deployed via Ollama on a $400 M2 Mac mini or via a $200/month A100 spot instance, processes the same queries at negligible incremental cost. The 7-point accuracy gap translates to 3,500 fewer correct queries per month — but those failures can be caught by an execution-based fallback: retry failed queries with GPT-4o only when execution fails. This hybrid approach achieves 97%+ of GPT-4o's quality at roughly 20% of the cost.

Beyond cost, our model has a deeper advantage specific to your platform: it is fine-tunable on your company's proprietary schemas. GPT-4o cannot see your internal table structures, proprietary time-series metrics, or custom TimescaleDB continuous aggregates without expensive API calls per query. Our model can be fine-tuned in one weekend on your internal query history, adapting to your column naming conventions, custom aggregate functions, and schema relationships. After this adaptation, it will outperform GPT-4o on your specific schemas — the exact use case your customers need. The 7-point gap is not a ceiling; it is a starting point that your team controls and can close through continued fine-tuning.
