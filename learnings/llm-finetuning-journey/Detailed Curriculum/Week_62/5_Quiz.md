# Week 62 Quiz — Head-to-Head Evaluation

## Multiple Choice

**Q1.** GPT-4o scores 83% on your custom benchmark. Your model scores 76%. You then test ONLY the TimescaleDB subset (50 examples) and find your model scores 70% while GPT-4o scores 65%. Which statement is most accurate for your technical report?

A) Your model is worse overall (76% < 83%) and you should not claim a win anywhere.
B) Your model is overall worse but domain-superior on TimescaleDB — a compelling result for your target use case.
C) The TimescaleDB subset result is not valid because it has only 50 examples.
D) You should retrain to close the overall gap before publishing results.

---

**Q2.** SQLCoder-7B was specifically designed and trained for SQL generation. Your model is also 7B and also SQL-specialized. What is the most informative comparison point between the two?

A) Parameter count and architecture — same architecture means direct performance comparison is fair.
B) Training data composition — the key differentiator is domain focus (general SQL vs. PostgreSQL/TimescaleDB).
C) Inference speed — smaller models of the same size should have identical speed.
D) BIRD-SQL accuracy — this is the only fair neutral-ground benchmark.

---

**Q3.** You cache GPT-4o API responses with MD5 hash of (prompt + system_prompt). A colleague says you should re-run GPT-4o without the cache to verify reproducibility. Is this necessary?

A) Yes — GPT-4o is non-deterministic (temperature=0 still produces non-identical outputs due to parallel sampling).
B) No — at temperature=0, GPT-4o's outputs are deterministic for identical inputs; caching is valid.
C) Yes — the MD5 hash might collide, invalidating the cache.
D) No — API reproducibility is guaranteed by OpenAI's SLA.

---

**Q4.** Your cost analysis shows: your model at $0/query (local), GPT-4o at $0.03/query. A user needs 10,000 correct queries per month. At equal accuracy, which is more cost-effective? At 76% vs 83% accuracy, what is the break-even cost per GPT-4o query at which the two models cost the same per correct query?

---

## Short Answer

**Q5.** You find that your model beats GPT-4o on 18 of the 50 TimescaleDB examples (36%) but GPT-4o beats you on 9 (18%), with 23 tied (both correct). Is your model statistically significantly better than GPT-4o on the TimescaleDB subset? Use McNemar's test framework (wins vs losses).

---

**Q6.** Your error analysis shows 40% of your failures are Type B (correct schema, wrong SQL logic). Surprisingly, GPT-4o also fails on 30% of the same questions (also Type B failures). What does this tell you about the nature of those questions?

---

## Deep Scenario

**Q7.** You are presenting your results to a potential employer (a startup building a PostgreSQL analytics platform). Your model achieves 76% on your custom benchmark vs GPT-4o's 83%. But your model: (a) runs locally at zero inference cost, (b) can be fine-tuned on the company's specific schemas, (c) processes queries in <2 seconds locally vs. 3–5 seconds via API. GPT-4o: (a) costs $0.03/query, (b) cannot be fine-tuned easily, (c) has higher general capability.

Write a 2-paragraph technical pitch explaining why your model is the right choice for this company's use case, while being honest about the accuracy gap.
