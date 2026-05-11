# Week 73 Answers

## Q1

**Answer: B**

**Why correct:** The geometric argument for superposition depends on sparsity. If in any given input only K out of F features are active simultaneously (where K << F), then two features that are "colocated" in the same neuron direction rarely interfere with each other — because most of the time only one of them is active. This is the mathematical foundation: the interference (dot product between feature directions) is small in expectation if the features are rarely simultaneously active. Remove sparsity — if all features were dense/always active — and superposition becomes catastrophic because all the stored features would interfere with each other simultaneously.

**Why others are wrong:**
- A: Redundancy (near-duplicate inputs) reduces the effective diversity of training data; it is not the mechanism that enables superposition.
- C: Low-rank covariance describes a specific structure but not the sparsity property that enables superposition.
- D: Normalization is a preprocessing concern; it does not enable more features than neurons.

---

## Q2

**Answer: B**

**Why correct:** Monosemanticity does not require activating on a single surface form — it requires activating on a single underlying concept. The feature activates on three different syntactic forms (Python `def`, SQL `CREATE FUNCTION`, JavaScript `function`) that all share the same abstract meaning: "this is where a function is being defined." This is a monosemantic feature for the concept "function definition," expressed across multiple programming languages. This is actually one of the most encouraging findings in interpretability — models appear to learn abstract, language-independent concepts rather than purely surface-form patterns.

**Why others are wrong:**
- A: Polysemanticity means responding to genuinely unrelated concepts; "function definition" in different languages is the same concept.
- C: Superposition refers to multiple features sharing neuron space; this SAE feature has already been separated from others.
- D: Multi-language activation is a feature, not a failure of training.

---

## Q3

**Answer: C**

**Why correct:** Activation patching is a causal intervention: you change the activations at a specific location to those from a different example, and observe whether the output changes. When patching head 15.4's values from a correct example causes the hallucination to disappear, this establishes that head 15.4's values at runtime causally contribute to the hallucination — changing them changes the output in the predicted direction. This is stronger evidence than correlation (high attention weight) but does not establish sufficiency: other heads may also be necessary for correct schema reading.

**Why others are wrong:**
- A: Sufficiency would mean head 15.4 alone can produce correct output; the patch just shows causal contribution.
- B: Necessity AND sufficiency requires ablation (head 15.4 alone is sufficient) plus intervention (removing head 15.4 causes failure).
- D: The patch came from a different example; memorization is not the explanation.

---

## Q4

**Answer: B**

**Why correct:** The logit lens shows that at layers 1–18, the residual stream already contains enough information to predict the correct column name. At layer 19, something (an attention head, an MLP, or their combination) modifies the residual stream in a direction that changes the top prediction. This is a strong signal to investigate layer 19's contribution: apply activation patching to zero out layer 19's attention output or MLP output and see if the correct prediction returns. This localizes the failure to a specific computational step.

**Why others are wrong:**
- A: Layers 1–18 are actively building the correct prediction; they are not irrelevant.
- C: There is no training bug implied; this pattern is how models compute — building up predictions and then updating them.
- D: More layers could help but the existing evidence does not require this conclusion.

---

## Q5

**Model answer:** High attention weight on a schema token shows correlation: when generating the next SQL token, the model looks at (attends to) the schema token with high weight. But the causal question is different: does reading that schema token actually affect what token is generated? Attention weight can be high due to positional biases or head specialization patterns that are independent of the token's content. The technique that establishes causality is activation patching: replace the key or value vector at the schema token position with values from a different example (e.g., one with a different column name). If the model's next-token prediction changes to match the new column name, the attention to that position is causally responsible for copying the column name. If the prediction does not change, the high attention weight was decorative.

---

## Q6

**Model answer:** Step 1: Attention visualization. For each of the 7 schema-hallucination failures (21% of 34 failures = ~7 examples), visualize the attention patterns at the final layer when generating the hallucinated token. Identify whether the model is attending to the wrong schema position (a different column), the question text (copying a word from the question), or a random position. This gives you the proximal cause: is the model looking at the wrong place, or looking at the right place but generating the wrong token? Step 2: Logit lens. For each failure, apply the logit lens to find at which layer the model commits to the hallucinated column name. If the hallucination commits early (layer 10–15 of 32), the failure is established in the middle layers before the attention heads that read the schema have had time to override it — suggesting the model's parametric memory is overriding the in-context schema. If it commits late (layer 28–32), an attention head in the final layers is responsible. The two steps together give you: what the model is attending to (Step 1) and when it decides to hallucinate (Step 2).

---

## Q7

**Model answer:** The most important SQL-safety features to find and suppress: (1) A "DDL/DML intent" feature that activates when the model is about to generate `DELETE`, `DROP`, `UPDATE`, or `INSERT` in response to a query framed as a SELECT question. Suppressing this feature during inference would prevent the model from generating data-modifying SQL regardless of how the question is phrased. (2) A "column invention" or "schema hallucination" feature that activates just before the model generates a column name not present in the provided schema. Suppressing this during inference would force the model to only use schema-grounded column names, reducing hallucination. Both of these are speculative — they may or may not exist as distinct monosemantic SAE features — but they are the highest-value targets for a SQL safety application.

---

## Q8 — Deep Scenario

**Model answer:** Option (a) — add JOIN training examples and retrain: Reliable if executed well (more JOIN examples will shift the model's distribution). Implementation complexity: low if you have the SFT pipeline from Week 58 ready. Compute cost: 1–2 GPU-hours for a targeted SFT run. Risk: you need to know which specific JOIN type distribution to target — if wrong JOIN is context-dependent (LEFT vs INNER depends on nullable foreign keys), more examples may not help without better structured examples.

Option (b) — activation steering: Suppress a specific direction at layer 22 corresponding to the wrong JOIN feature. Not yet reliable for production: activation steering can have unintended side effects (suppressing the wrong JOIN feature may degrade correct JOIN generation). Implementation complexity: high (requires finding the specific direction via SAE or probing). Compute cost: low at inference time but high to develop reliably.

Option (c) — post-processing SQL validator: Parse generated SQL, detect wrong JOIN type using schema context, and re-run generation. Reliable if the validator catches the right errors. Implementation complexity: medium (requires a SQL parser and schema-aware JOIN validation logic). Compute cost: 1.5–2x inference time for re-generation on failed queries.

Recommendation: Option (a) + (c) in combination. Add 200 targeted JOIN examples (specifically nullable foreign key joins) and re-run SFT — this costs 30 minutes of compute and directly addresses the root cause. Simultaneously deploy the SQL validator as a safety net for the cases your retraining misses. Option (b) is interesting research but not ready for production use with your current compute budget.
