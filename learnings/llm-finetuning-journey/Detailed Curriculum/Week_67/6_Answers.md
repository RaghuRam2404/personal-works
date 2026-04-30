# Week 67 Answers

## Q1

**Answer: A**

**Why correct:** Data contamination in evaluation means that test examples appeared in the training data, inflating the apparent accuracy. For a benchmark you created yourself, the key risks are: (1) you wrote evaluation examples after seeing training data, allowing unconscious overlap, and (2) evaluation examples were accidentally included in the training set. The paper must document the timeline (benchmark examples were written before or independently of training data), provide a deduplication check (no benchmark example has >0.8 ROUGE-L overlap with any training example), and ideally commit benchmark examples to a public repo with a date stamp that predates training.

**Why others are wrong:**
- B: Sampling bias is a separate concern from contamination; contamination refers specifically to train-test overlap.
- C: Metric validity is a legitimate but separate concern.
- D: Evaluation bugs are also a separate concern.

---

## Q2

**Answer: B**

**Why correct:** When a student model (your fine-tuned Qwen) is trained on outputs from a teacher model (GPT-4o-mini), it can inherit systematic errors, biases, or hallucinations from the teacher. For SQL specifically, GPT-4o-mini may generate syntactically valid but semantically incorrect queries that pass simple format checks. These propagate into the student's learned behavior. A reviewer expects you to address this with: (1) your LLM-as-judge filtering pass (which catches wrong SQL), and (2) your human-curated 1,247 examples as a quality anchor. Cite the risks of synthetic data and explain your mitigations.

**Why others are wrong:**
- A: OpenAI's terms of service prohibit using outputs to train competing models, but this is a licensing/legal issue, not a technical one; most academic preprints use synthetic data regardless.
- C: LIMA showed 1,000 examples of high-quality diverse general instruction following — SQL is a specialized domain where more examples genuinely help.
- D: GPT-4o-mini is capable of generating correct SQL for most Spider/BIRD patterns.

---

## Q3

**Answer: B**

**Why correct:** The problem motivation ("text-to-SQL is hard, TimescaleDB is underserved") is the most cuttable content in an abstract because: (a) the venue's readers already know why text-to-SQL matters, (b) the specific domain claim (TimescaleDB) can be implied by your benchmark name rather than stated explicitly, and (c) the motivating sentence contributes the fewest bits of unique information. Numbers (A) are irreplaceable — they are the entire reason someone reads an abstract. Released artifacts (C) are concisely expressed in one phrase. Methodology (D) is essential for understanding what makes your approach different.

**Why others are wrong:**
- A, C, D: All carry high-density, unique information that cannot be recovered from other parts of the abstract.

---

## Q4

**Answer: B**

**Why correct:** LLM benchmark results must be versioned because frontier models change frequently and same-name models can have significantly different performance across versions. Reporting "GPT-4o (gpt-4o-2024-11-20, evaluated December 2024)" allows readers to understand exactly which model capability you measured and to contextualize the result relative to subsequent model updates. This is now standard practice in LLM papers. The note also signals to readers that your comparison may be outdated, which is honest and useful.

**Why others are wrong:**
- A: Removing a comparison weakens your paper; versioned comparisons are valuable.
- C: Switching to Claude because it is "more reproducible" is not a principled reason.
- D: Averaging across different model versions is scientifically wrong — they are different models.

---

## Q5

**Model answer:** A contribution is something you built or did that adds to the field — it is a deliverable. A finding is something you observed through experimentation — it is a result. Contribution example: "We release a 200-example TimescaleDB evaluation benchmark with expert-written SQL." This is a thing you made that others can use. Finding example: "GRPO with executable rewards improved exact-match accuracy by 4.2 pp over DPO-only on our benchmark." This is what you learned by running an experiment. A technical report should have 3–6 contributions and 5–15 findings. Confusing them weakens both — presenting a finding as a contribution sounds like you are overclaiming; presenting a contribution as a finding undersells what you made.

---

## Q6

**Model answer:** The four specific details a reproducibility reviewer needs: (1) Which LLM was used as the judge, at what API version (e.g., "GPT-4o-mini, gpt-4o-mini-2024-07-18"). (2) The exact scoring prompt — either the full text or a reference to Appendix X where it is printed. (3) The score threshold used for acceptance (e.g., "score ≥ 4 on a 1–5 scale"). (4) The acceptance rate (e.g., "67% of generated examples passed"). Without all four, another researcher cannot reproduce the filtered dataset from the raw generated outputs.

---

## Q7

**Model answer:** Handle this by being explicit about what you measured and what you did not. Write something like: "SQLCoder-7B achieves 80.2% on Spider 1.0 dev (published, Defog 2023). We did not run SQLCoder-7B on our TimescaleDB benchmark; we report only models we evaluated ourselves on the same test set under identical conditions." Then in your results table, mark SQLCoder-7B's Spider score as "(published)" and mark your model's Spider score as "(ours)" to make the comparison apples-to-apples within each benchmark. Do not present published scores on a different benchmark alongside your own scores on a different benchmark in the same table row — this is a common and misleading error.

---

## Q8 — Deep Scenario

**Model answer:** The most likely cause is that the reported 25,500 figure was the count before a final deduplication pass that ran during the dataset export to HuggingFace — removing 683 near-duplicate examples. Alternatively, the number in the paper came from a W&B log mid-run, and a subsequent filtering step reduced the count before the final push.

Required actions: Issue a v2 of the preprint with the corrected number (24,817) and a note in the changelog. Update the dataset README to reflect the same count. Do not change the number without a visible correction notice.

Future prevention process: At paper submission time, run a single script (`audit_datasets.py`) that counts examples in every dataset file and asserts against the number in the paper's LaTeX source. Keep this script in the repo and run it as a CI check before any preprint update. Numbers in papers must come from code, not from memory or intermediate logs.
