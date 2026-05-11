# Week 71 — Frontier Reading 1: Tulu 3, SmolLM2, OLMo 2

## Learning Objectives

By the end of this week, you will be able to:

- Summarize the key training innovations in Tulu 3 (RLVR, on-policy data generation) and relate them to your own GRPO training
- Explain how SmolLM2 achieves strong performance at 135M–1.7B parameters and what training decisions drive this
- Describe OLMo 2's fully open training stack and what it reveals about the relationship between data quality and model capability
- Identify at least three techniques from these papers that you could apply to improve postgres-sqlcoder-7b
- Synthesize across the three papers: what is the emerging consensus in 2024–2025 instruction tuning?

## Concepts

### How to Read a Frontier LLM Paper

Reading an LLM technical report is a skill. You are not reading for full comprehension of every section — you are reading to extract: (a) what they did that is new, (b) what numbers demonstrate the improvement, (c) what is reusable for your own work. A structured reading approach:

1. Read abstract + introduction (15 min): identify the 3 claimed contributions.
2. Read the evaluation section (15 min): find the key results table, identify which benchmarks matter.
3. Read the dataset section (20 min): this is where reusable ideas live.
4. Read training section (20 min): extract hyperparameters and novel loss functions.
5. Skim related work and limitations (10 min).
6. Write your synthesis notes (30 min).

Total: 90 minutes per paper. With three papers this week, that is 4.5 hours of active reading.

### Tulu 3: On-Policy RLVR for Open Models

Tulu 3 (Ivison et al. 2024) is the successor to Tulu 2 and represents one of the most complete public instruction-tuning pipelines available. Its key contribution is RLVR (Reinforcement Learning with Verifiable Rewards) applied to general instruction following, not just math or code.

Key Tulu 3 innovations to understand:

First, on-policy data generation during SFT. Rather than using a fixed static dataset, Tulu 3 generates new training examples on-the-fly using the model itself — a form of self-improvement. For SQL, this is directly applicable: you can generate SQL from your trained model, verify executability, and add passing examples back to the training set.

Second, RLVR with diverse reward types. Tulu 3 uses verifiable rewards (math, code execution, factual) rather than a learned reward model. This is precisely what your GRPO training does — executable-SQL verification is a verifiable reward. Tulu 3's success validates this approach at scale.

Third, preference data quality over quantity. Tulu 3 uses 326K preference pairs, but with careful curation. The key finding: 10K high-quality pairs outperform 100K low-quality pairs in DPO. This retroactively justifies your 5K curated preference dataset over generating 50K noisy pairs.

### SmolLM2: Efficient Small Models at 135M–1.7B

SmolLM2 (Allal et al. 2024) demonstrates that very small models (sub-2B parameters) can achieve competitive performance on targeted tasks when trained on high-quality curated data.

Key SmolLM2 findings relevant to your work:

Data quality beats scale at the small-model tier. SmolLM2-1.7B on math benchmarks outperforms models twice its size trained on lower-quality data. For your SQL use case, this validates the argument that a 7B model trained on your curated 25K examples can compete with much larger models trained on generic data.

The FineWeb-Edu / DCLM curriculum: SmolLM2 uses staged curriculum data — starting with high educational-quality web text, then narrowing to domain data. The analogy for your work: CPT on PostgreSQL documentation before SFT on SQL pairs is the small-model analogue of this curriculum.

Tokenizer efficiency: SmolLM2 uses an optimized tokenizer with better SQL and code token coverage than older tokenizers. For SQL models, tokenizer efficiency matters because SQL keywords like `time_bucket` or `generate_series` may be split into multiple tokens, increasing effective sequence length.

### OLMo 2: The Fully Open Training Stack

OLMo 2 (Groeneveld et al. 2024) is significant not for a single algorithmic contribution but for its commitment to full openness: training data (Dolmino Mix), training code (OLMo), evaluation harness (OLMES), and intermediate checkpoints are all released.

Key OLMo 2 findings:

Mid-training data mixing: OLMo 2 uses a two-phase pretraining strategy where the second phase overweights high-quality data (StackExchange, code, Wikipedia) even though the first phase was more diverse. The final-phase data quality improvement accounts for 2–4 pp on downstream tasks. This directly explains why your CPT on PostgreSQL-specific data (the "second phase" in your pipeline) helps more than random text.

Staged SFT with decreasing temperature: OLMo 2's SFT uses a temperature that decreases from 1.0 to 0.0 over training — starting with diverse exploration and ending with confident, low-temperature outputs. You used temperature=1.0 throughout; the staged approach might further improve SQL generation consistency.

The open stack as research infrastructure: Because OLMo 2 releases intermediate checkpoints, researchers can study what each phase contributes. This is the gold standard for reproducible LLM research and validates the approach you took in your technical report (ablation study with intermediate checkpoints).

### Synthesis: The 2024 Instruction Tuning Consensus

Reading these three papers together reveals an emerging consensus:

1. Data quality > data quantity: all three papers find that careful curation and filtering beats raw scale.
2. Verifiable rewards work: RLVR/GRPO with executable rewards outperforms RLHF with learned reward models for tasks with objective correctness criteria (code, math, SQL).
3. Curriculum matters: staged training (CPT → SFT → alignment) with increasing specificity outperforms one-stage training.
4. On-policy data generation improves alignment: using the current model to generate training data creates a self-reinforcing improvement loop.
5. Open models are closing the gap: SmolLM2-1.7B and OLMo 2-7B match or exceed GPT-3.5 on targeted tasks.

## Connections

These papers validate and extend the decisions you made in Weeks 53–60. Tulu 3's RLVR directly aligns with your GRPO training (Week 60). SmolLM2's data quality findings vindicate your LLM-as-judge filtering (Week 55). OLMo 2's mid-training data mixing strategy contextualizes your CPT design (Week 57). Reading these papers now — after building the full pipeline — gives you the theoretical vocabulary to describe what you built.

Weeks 72–74 continue the frontier reading arc with DeepSeek/Qwen, Anthropic interpretability, and context extension papers.

## Common Misconceptions / Pitfalls

The most common mistake in reading papers is passive reading — absorbing without questioning. For each claim, ask: "Is this measurement on the same task as mine? Would this apply to a SQL-specialized 7B model?" Not every finding transfers to your use case.

Do not try to read all three papers at full depth. Use the structured reading approach: 90 minutes per paper, extraction-focused.

## Time Allocation (6–8 hours)

- 1.5h: Read Tulu 3 (focus: RLVR section and data ablations)
- 1.5h: Read SmolLM2 (focus: data curriculum and evaluation results)
- 1.5h: Read OLMo 2 (focus: mid-training data mixing and open infrastructure)
- 1.5h: Write synthesis notes in `reading_notes/week71_synthesis.md`
