# Week 76 Quiz — Multi-Turn Agentic SQL with Tool Use

Difficulty: Senior research engineer. Questions require reasoning about multi-turn training dynamics, tool-use formats, and agentic evaluation design.

---

## Multiple Choice

**Q1.** You implement multi-turn SFT on agentic trajectories and forget to mask loss on the tool-result turns (i.e., you compute cross-entropy loss on the database execution results as if they were model outputs). What is the most likely observable effect?

A. Training loss decreases faster because the model has more tokens to learn from.
B. The model learns to predict database outputs rather than SQL queries, and at inference time emits text that resembles PostgreSQL row output instead of tool calls.
C. Training diverges immediately because tool-result tokens are out-of-distribution for the model's vocabulary.
D. No effect — the model ignores tool-result tokens because they follow a different formatting pattern.

---

**Q2.** You evaluate your agentic model and find: first-attempt EM = 81%, final EM = 83.5%, correction rate = 28%. Your single-shot baseline (same model, no agentic loop) achieves 83.1% EM. The most accurate interpretation is:

A. The agentic model is strictly better because final EM (83.5%) exceeds the single-shot baseline (83.1%).
B. The agentic model underperforms because first-attempt EM (81%) is lower than the single-shot baseline, suggesting agentic SFT degraded single-shot capability.
C. The result is ambiguous: first-attempt degradation suggests the model may be "holding back" in round 1, but the net gain after correction is marginal; you need to compare on more examples and measure latency cost.
D. The correction rate (28%) is too low; a well-designed agentic model should correct 90%+ of its mistakes.

---

**Q3.** Your agentic loop is configured with `max_rounds=3`. On 12% of test examples, the model reaches 3 rounds without producing a final answer (it keeps emitting tool calls). The root cause is most likely:

A. The model has too low a temperature, causing it to repeat the same query.
B. The model was trained on trajectories that always had exactly 2 correction rounds, so it learned that 2 tool calls must precede every final answer.
C. The PostgreSQL connection is slow, causing the model to time out and re-emit tool calls.
D. The model's context window is exceeded after 3 rounds, causing it to loop.

---

**Q4.** You want to evaluate whether the improvement from the agentic loop comes from (a) the self-correction training or (b) simply having access to execution feedback at inference time. The correct ablation is:

A. Compare your agentic-SFT model (with loop) to your single-shot model (without loop).
B. Compare your agentic-SFT model (with loop) to your single-shot model (also with loop, but no agentic training).
C. Compare your agentic-SFT model (with loop) to a GPT-4o baseline.
D. Compare your agentic-SFT model (with loop) to your agentic-SFT model without the tool-result turn (no database feedback).

---

**Q5.** In the ReAct (Reason + Act) formulation for agentic SQL, the "Reason" step is a natural-language thought the model produces before emitting a tool call. When would you include the Reason step in your training data, and when would you omit it?

A. Always include it — reasoning always improves tool call accuracy.
B. Always omit it — reasoning traces increase latency and are not worth the trade-off for SQL.
C. Include it for complex queries (multi-table JOINs, subqueries) where the model needs to decompose the problem; omit it for simple queries to reduce token overhead.
D. Include it only if your base model was pretrained with chain-of-thought data; reasoning traces harm models without this pretraining.

---

## Short Answer

**Q6.** Explain the difference between agentic SQL training (SFT on multi-turn trajectories) and agentic SQL inference (running any model in an agentic loop). Under what conditions does training on agentic trajectories provide meaningful improvement over simply running a single-shot model inside a loop?

---

**Q7.** You are building the trajectory dataset using synthetic error injection. You apply three mutation types: drop JOIN, wrong column name, syntax error. A colleague points out that real model failures are different from these synthetic mutations — real models tend to fail on semantic errors (selecting the wrong aggregation level, missing a WHERE filter) rather than syntax errors. How does this mismatch between synthetic training errors and real inference errors affect your model, and what would you do to address it?

---

**Q8.** Your agentic loop has access to execution results (database rows). Describe two concrete ways you would use the returned rows — not just error messages — to provide feedback to the model. For each, give a specific SQL scenario where row-level feedback catches an error that error-message feedback would miss.

---

## Deep Scenario

**Q9.** You deploy your agentic SQL model in production. The system receives a question: "Show me the top 5 customers by revenue in the last 30 days, broken down by product category." Your model emits the following sequence:

Round 1: `SELECT customer_id, SUM(amount) FROM orders WHERE date > NOW() - INTERVAL '30 days' GROUP BY customer_id ORDER BY SUM(amount) DESC LIMIT 5;` — executes successfully, returns 5 rows.

Round 2 (model decides to refine): `SELECT c.name, p.category, SUM(o.amount) FROM orders o JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id WHERE o.date > NOW() - INTERVAL '30 days' GROUP BY c.name, p.category ORDER BY SUM(o.amount) DESC LIMIT 5;` — executes successfully, returns 5 rows.

The model emits round 2 even though round 1 "worked." Write a structured analysis that: (a) explains why this behavior is beneficial (what did the model learn to do?), (b) identifies the risk this behavior creates in production, (c) proposes one training-time intervention and one inference-time intervention to ensure the model stops at the right round.
