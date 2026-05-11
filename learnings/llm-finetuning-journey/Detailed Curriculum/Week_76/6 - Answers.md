# Week 76 Answers — Multi-Turn Agentic SQL with Tool Use

---

## Q1. Answer: B

**Why B is correct:** If loss is computed on tool-result tokens (which contain database row data — numbers, strings, column names), the model's training objective becomes "predict database outputs given the conversation so far." Over hundreds of steps, this pulls the model toward generating text that resembles query results rather than SQL queries. At inference time, this manifests as the model emitting row-like text in positions where it should emit tool calls — a subtle failure mode that can be hard to detect without inspecting raw generations.

**Why A is wrong:** More tokens do reduce loss faster in a narrow sense, but the model is learning the wrong thing. Lower loss on an incorrect objective is strictly worse than higher loss on the correct one.

**Why C is wrong:** Tool-result tokens (numbers, words, punctuation) are entirely in-vocabulary for any LLM. There is no out-of-distribution vocabulary issue.

**Why D is wrong:** The model does not independently ignore tokens based on formatting. It learns from every token that is not masked, regardless of structural position.

---

## Q2. Answer: C

**Why C is correct:** Both A and B capture part of the truth but miss the full picture. Final EM (83.5%) does beat the single-shot baseline (83.1%), which is a positive signal. But first-attempt EM (81%) is lower than the baseline — meaning the agentic SFT has reduced the model's ability to get it right in one shot. This is a known trade-off: training on agentic trajectories can slightly harm single-shot performance because the model learns to "expect" a correction round. The net gain is only 0.4 pp after the full loop, which may not justify the 2x latency increase. More data (ideally 200 examples, not 40) and a confidence interval are needed before claiming improvement.

**Why D is wrong:** A 28% correction rate means the model successfully corrects 28% of its initial failures. This is actually a reasonable number — it does not mean the model should be expected to correct 90%+ of errors. Complex semantic errors are legitimately hard to fix from error messages alone.

---

## Q3. Answer: B

**Why B is correct:** If every trajectory in your training data has exactly two correction rounds before the final answer, the model learns the pattern "emit two tool calls, then emit a final answer." At inference time, it applies this learned distribution regardless of whether two corrections are actually needed. The fix is to vary the number of correction rounds in your training data: include examples with zero corrections (model gets it right in round 1), one correction, two corrections, and rare three-correction cases.

**Why A is wrong:** Temperature affects output diversity, not the structural pattern of when to stop emitting tool calls.

**Why C is wrong:** Connection latency affects wall-clock time, not the model's generation behavior. The model does not observe connection latency.

**Why D is wrong:** After 3 rounds, the context is longer but unlikely to exceed the context window for a 2048-token context with brief database results. Context overflow manifests as truncation or incoherence, not persistent looping.

---

## Q4. Answer: B

**Why B is correct:** The correct ablation isolates the training variable. Both models (agentic-SFT and single-shot) run inside the same agentic loop at inference time. If agentic-SFT beats single-shot-in-loop, the improvement comes from training on agentic trajectories. If they perform similarly, the loop itself is sufficient and agentic SFT adds no value. This is the cleanest experimental design.

**Why A is wrong:** This conflates two variables: training procedure and inference procedure. You cannot tell whether the improvement comes from agentic training or agentic loop if only one model has access to the loop.

**Why C is wrong:** A GPT-4o comparison measures the gap to a frontier model, not the contribution of agentic training specifically.

**Why D is wrong:** Removing the tool-result turn tests a different question (does the model need execution feedback at all?) rather than whether agentic training matters.

---

## Q5. Answer: C

**Why C is correct:** The Reason step has real cost (additional tokens, additional latency) and real benefit (better query decomposition for complex queries). The correct engineering decision is to include reasoning for complex examples in your training data (annotated with a complexity flag) and omit it for simple ones. At inference time, you can route queries to either template based on a lightweight classifier (schema complexity, question length, presence of aggregation keywords). This avoids paying the reasoning overhead on the 40% of queries that are straightforward single-table lookups.

**Why A is wrong:** Reasoning is beneficial for complex cases but adds noise and latency for simple ones. Always including it is a poor default.

**Why B is wrong:** Reasoning traces have been empirically shown to improve SQL generation on multi-table benchmarks. Omitting them entirely leaves performance on the table for hard queries.

**Why D is wrong:** This is too restrictive. Models without explicit CoT pretraining (like Qwen2.5-Coder base) still benefit from reasoning traces when fine-tuned on them, because the traces provide a structured decomposition that guides SQL generation.

---

## Q6 — Short Answer

Agentic SQL training means including multi-turn execution-feedback trajectories in your SFT dataset. The model learns the pattern: emit tool call → observe error → revise query. Agentic SQL inference means taking any model and placing it inside a loop that executes its SQL and feeds results back, regardless of whether the model was trained on such trajectories.

The key question is: does training on agentic trajectories help beyond simply using the loop at inference time? The answer is yes, under two specific conditions. First, when the model needs to interpret structured error messages to diagnose what went wrong — e.g., a `psycopg2.errors.UndefinedColumn` error implies a specific fix (column name correction), and a model trained on such error-correction examples learns to parse and act on this signal more effectively than a model that has never seen it. Second, when the model must distinguish between "the query ran and returned rows" (stop) and "the query ran but the rows look wrong based on question semantics" (continue and revise) — the latter requires learned judgment that single-shot training does not develop.

---

## Q7 — Short Answer

The mismatch between synthetic training errors (syntax errors, dropped JOINs) and real model failures (semantic errors — wrong aggregation, missing WHERE filter) means your model learns to self-correct the wrong failure modes. At inference time, when the model produces a semantically wrong query that executes without error (returns rows, but the wrong rows), the model's trained correction behavior does not activate — because it was only trained to respond to error messages, not to semantically wrong results. The model will accept the incorrect result and emit it as a final answer.

To address this: first, include "silent failure" trajectories in your training data — examples where round 1 executes successfully but produces the wrong answer, the tool result contains a validation signal (e.g., "0 rows returned" when the question implies there should be results, or a row count that contradicts expectations), and the model learns to ask follow-up questions or revise based on result shape. Second, generate real model failure trajectories by running your actual model on your training set, collecting its incorrect generations, and using those (not synthetic mutations) as round-1 queries. This aligns training errors with inference errors.

---

## Q8 — Short Answer

First, row count as a signal: if the question asks "find all customers who purchased more than $1000" and the executed query returns 0 rows, this is a strong signal that a filter is too restrictive or a table name is wrong. You can add a check: `if len(rows) == 0 and question implies non-empty result → feed "Query returned 0 rows. This may indicate a WHERE clause error." back to the model.` This catches cases where `WHERE amount > 100000` (wrong scale) silently filters out everything.

Second, schema mismatch detection: the model can compare the column names in the returned rows against the column names mentioned in the question. For a TimescaleDB query asking for "revenue by hour," if the returned rows contain a column named `amount_total` but the question used "revenue," this mismatch is a signal to check whether the model used the correct column. Feeding "Returned columns: amount_total, hour_bucket — does this match 'revenue by hour'?" gives the model information it would not have from the error message alone.

---

## Q9 — Deep Scenario

**Model answer:**

The model's behavior in round 2 is beneficial because it demonstrates genuine semantic understanding of the question. Round 1 answered a simpler version of the question (top 5 customers by total revenue), but the model recognized that "broken down by product category" requires an additional JOIN and GROUP BY dimension. This is evidence that the model is using the result from round 1 as a semantic anchor — it sees a valid but incomplete result and refines toward the full answer. This is exactly the agentic behavior you wanted to train.

The production risk is over-refinement: the model may continue issuing tool calls even when the current result is fully correct, driven by a bias in the training data where more rounds always led to better results. This creates: (a) unnecessary latency, (b) a risk that round 2 introduces a bug into a previously correct query, and (c) non-deterministic behavior where the same question produces different final SQL on different runs.

Training-time intervention: include a "stop early" signal in your trajectories. When the round-1 result is semantically correct (i.e., it matches the gold answer), the model's round-2 turn should simply emit the round-1 SQL as its final answer, not a new tool call. Label 30–40% of your training examples as "round-1 correct, stop immediately" so the model learns the stopping criterion.

Inference-time intervention: implement an EM guard at the orchestration layer. Before feeding round-N results back to the model, execute a lightweight semantic correctness check (e.g., compare result column names against required columns from the question, or compare row count against expected cardinality). If the check passes, short-circuit the loop and return the current SQL without asking the model to continue.
