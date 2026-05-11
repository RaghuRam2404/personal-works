# Week 70 Answers

## Q1

**Answer: B**

**Why correct:** B is specific (which databases), testable (accuracy can be measured on MySQL/Snowflake), and mechanistic (explains why — "dialect-specific syntax differences in date arithmetic and window functions"). It gives a reader exactly the information they need to scope their use of your model. Options A, C, and D are vague, deflecting, or vacuous.

**Why others are wrong:**
- A: "May not work well" and "all SQL databases" are both unspecified — no reader can act on this.
- C: "Future work will address" is future work, not a current limitation.
- D: "Limitations like all ML systems" is meaningless — it applies to everything.

---

## Q2

**Answer: C**

**Why correct:** "Partial" with a clarifying footnote is the honest response. Optimizer epsilon and beta2 are indeed widely-used defaults, but "widely known" does not mean "reported." The NeurIPS checklist exists precisely to ensure that defaults are not assumed — a researcher using a different optimizer library may have different defaults. Adding "we used standard defaults: β1=0.9, β2=0.999, ε=1e-8" takes one sentence and makes your training fully specified. Answering "Yes" when you have not reported these values is inaccurate.

**Why others are wrong:**
- A: The checklist applies to all ML papers including fine-tuning papers.
- B: Answering Yes when the answer is Partial misrepresents the paper's reproducibility.
- D: Answering No without explanation is worse than answering No with a footnote.

---

## Q3

**Answer: B**

**Why correct:** arXiv requires an endorsement from an existing arXiv author in the relevant category (cs.CL or cs.LG) for first-time submitters without institutional affiliation. An endorser is a researcher who has previously published on arXiv in that category and agrees to endorse your submission. You can find endorsers through your professional network, by emailing authors of related work, or through the ML community on Twitter/X. This is a well-known process and is not prohibitively difficult for a quality submission.

**Why others are wrong:**
- A: arXiv accepts non-institutional authors with endorsement.
- C: There is no "arXiv Pro" paid tier.
- D: A co-author with institutional email helps but is not the required mechanism; endorsement works independently of co-author affiliation.

---

## Q4

**Answer: B**

**Why correct:** A good future work statement specifies: (1) what you would do (fine-tune Qwen2.5-72B with QLoRA r=16 at 4-bit), (2) what compute you expect to need (4× A100s), (3) what result you expect and why (5–8 pp improvement, based on Chinchilla scaling). This is a research plan, not a wish. A reviewer can evaluate whether this prediction is reasonable and whether the experiment is feasible.

**Why others are wrong:**
- A: "When compute becomes available" is a schedule note, not a research direction.
- C: "Will likely" without mechanism or evidence is speculation.
- D: Completely content-free.

---

## Q5

**Model answer:** "Our model is trained and evaluated on single-turn pairs (one question → one SQL query). We do not characterize performance on iterative refinement workflows where the user provides a previous query, an error message, and a follow-up correction request. Preliminary informal evaluation suggests multi-turn performance degrades significantly when context exceeds 2 turns; a systematic study using CoSQL or SParC as the evaluation benchmark would quantify this gap."

This version identifies: the specific capability missing (multi-turn iterative refinement), the practical context where it matters (iterative SQL development), and the experiment that would measure it (CoSQL/SParC evaluation).

---

## Q6

**Model answer:** The evaluation prompt template is the exact string passed to the model to generate SQL. Even small differences — a different section header, an extra newline, rephrased instructions — shift the conditional probability distribution of the model's output and change accuracy. For example, your training used the header `### SQL Query` as the generation trigger. A paraphrased template using `## SQL:` instead has different tokenization (one vs two `#` characters, a colon vs nothing) and produces a different distribution over first-token SQL keywords. In our own evaluation, we observed a 14 pp drop when the prompt format was changed from training format to a paraphrased version — the model was trained on the exact trigger and is sensitive to it. Publishing the verbatim template ensures other researchers can reproduce your numbers without discovering this sensitivity through trial and error.

---

## Q7

**Model answer (Twitter/X version):**

"postgres-sqlcoder-7b: a 7B open-weight SQL model for PostgreSQL + TimescaleDB, fine-tuned via CPT→SFT→DPO→GRPO. 83.1% on our domain benchmark — better than GPT-4o (79.4%). Runs locally in 4.5 GB. Code + model + data: [HF link] | Report: [arXiv link]"

(270 characters — fits in 280.)

---

## Q8 — Deep Scenario

**Model answer:**

Three possible explanations for the 7 pp gap: (1) Wrong model — the researcher may have loaded your SFT checkpoint rather than the final GRPO checkpoint, or loaded a quantized variant (AWQ INT4 at 82.6%) rather than BF16. (2) Wrong prompt format — the researcher may have used a different template than your evaluation script uses; the model is sensitive to the exact `### SQL Query` trigger. (3) Wrong evaluation script — the researcher may be computing exact-match without your normalization steps (lowercasing, whitespace collapsing), which would lower EM by 3–8 pp on typical SQL benchmarks.

First 24 hours: (a) Run your own evaluation fresh from the published release and record every step with screenshots. (b) Contact the researcher and ask for their evaluation script, model checkpoint path, and prompt template. (c) Check whether the Custom-200 benchmark file on HuggingFace matches the exact file you used internally (file hash comparison).

What to publish: Post a GitHub issue and an arXiv comment ("technical correspondence") within 48 hours, regardless of the root cause. If the error is in your paper (wrong checkpoint documented), issue a v2 with a correction notice. If the error is in the researcher's setup, publish a "reproducibility notes" document with exact commands to reproduce the 83.1% result step by step.
