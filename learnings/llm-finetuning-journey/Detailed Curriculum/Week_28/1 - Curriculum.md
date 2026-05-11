# Week 28 — What is Fine-Tuning, Really?

## Learning Objectives

By the end of this week, you will be able to:

- Distinguish continued pretraining, supervised fine-tuning (SFT), and instruction tuning — and explain when each is appropriate
- Describe the modern post-training pipeline: SFT → DPO → GRPO
- Explain why fine-tuning on a small labeled dataset can dramatically shift model behavior without destroying pretrained knowledge
- Read and understand InstructGPT's training methodology (sections 1–3)
- Map Karpathy's conceptual walkthrough of LLMs to the technical concepts you have built so far

---

## Concepts

### 1. The Three Flavors of Updating a Pretrained Model

You finished Phase 3 having trained transformer language models from scratch and having understood scaling laws. Now the question changes: given an already powerful pretrained model, how do you specialize it?

There are three distinct approaches, often confused:

**Continued pretraining (domain-adaptive pretraining, DAPT)**
You take a pretrained base model and continue running the standard next-token prediction loss on a large corpus in your target domain — no labels, no instruction format, just raw text. This is appropriate when the base model has seen little of your domain's vocabulary, syntax, or content. For example, if you want a model that understands PostgreSQL system catalogs or TimescaleDB hypertable internals, and the base model was trained mostly on generic code, continued pretraining on raw PostgreSQL documentation and query logs would shift the token distribution before you do any supervised step. The tradeoff: you need large amounts of domain text (10B+ tokens is common) and significant compute.

**Supervised fine-tuning (SFT)**
You have labeled (input, output) pairs and train the model with cross-entropy loss on the output tokens only (the input is masked). No unsupervised pretraining structure — just teacher-forcing on desired outputs. SFT is the step that turns a base model into a chat or instruction-following model. For your domain, SFT means: given a natural language question + schema, train the model to output the correct SQL. SFT works with surprisingly few examples (1K–100K) because the model already knows SQL syntax; you are adjusting which SQL patterns it activates for which inputs.

**Instruction tuning**
A specific flavor of SFT using prompt-response pairs formatted as instructions. InstructGPT (ChatGPT's ancestor) was instruction-tuned on a dataset of human-written (instruction, response) pairs curated to cover a wide variety of tasks. The distinction from vanilla SFT is mostly about the data format and diversity, not the loss function. After instruction tuning, a model follows natural language instructions rather than simply autocompleting.

### 2. When to Use Which

| Scenario | Method | Minimum data |
|---|---|---|
| Base model barely knows your domain's vocabulary | Continued pretraining | 1B–100B tokens |
| Base model knows the domain but needs task-specific behavior | SFT | 1K–100K examples |
| You want general instruction-following + your domain | Instruction tuning on mix | 10K–1M examples |
| You want to shape output style/helpfulness after SFT | DPO or GRPO | 1K–50K preference pairs |

For your PostgreSQL text-to-SQL project: `Qwen2.5-Coder-7B` was pretrained on large amounts of code including SQL, so continued pretraining is unlikely to help much. SFT on (question + schema → SQL) pairs is the right first step. You will do this starting Week 29.

### 3. The Modern Post-Training Pipeline

As of 2024–2025, state-of-the-art open models go through at least three stages after pretraining:

```
Base Model (pretrained)
       |
   Stage 1: SFT
   (instruction following, format learning)
       |
   Stage 2: DPO or RLAIF
   (preference alignment, helpfulness, harmlessness)
       |
   Stage 3: GRPO or RLVR
   (verifiable reward — e.g., math/code correctness)
       |
   Deployed Model
```

You will implement all three stages across Phase 4 (SFT) and Phase 5 (DPO, GRPO). This week is about understanding why this pipeline exists and what each stage contributes.

SFT teaches format and task. DPO refines between good and better outputs (ranking). GRPO trains on verifiable scalar rewards — for SQL, "does the query return the correct rows?" is an objective, binary reward. This is the pipeline that produced DeepSeek-R1 and the reasoning models of 2025.

### 4. Why Fine-Tuning Works: The Intrinsic Low-Rank Hypothesis

A key insight from the LoRA paper (which you will study in Week 30): when you fine-tune a large model, the actual weight changes (delta W) are low-rank. The model does not need to relearn everything — it only needs to shift a small subspace of its representations. This is why SFT on even 1K well-curated examples can produce dramatic behavioral change: you are not fighting against 7 billion pretrained parameters, you are nudging the activation subspace toward your target distribution.

This also explains why catastrophic forgetting is a real risk: if you fine-tune too aggressively (high learning rate, too many epochs), you overwrite more of the pretrained subspace than the task requires. LoRA mitigates this by constraining updates to a low-rank matrix.

### 5. Karpathy's Deep Dive: What to Pay Attention To

This week's main video is Karpathy's 3.5-hour walkthrough. It is dense and worth watching twice (or at 1x speed with pausing). Key sections to focus on:

- The pretraining → post-training distinction (first 30 minutes)
- How RLHF was discovered and why it works (around the 1h mark)
- The discussion of emergent capabilities and why they appear at scale
- His framing of base models as "document completers" vs. instruction-tuned models as "assistants"

You have now built the transformer from scratch and studied scaling laws. Karpathy's explanations should click at a much deeper level than they would have in Week 1.

---

## Connections

**Builds on:** Phase 3 pretraining (Weeks 17–27) — you now understand what the model learned during pretraining and what fine-tuning is changing. Scaling laws (Week 20) explain why a 7B model trained on 1T tokens is a better starting point than a 100M model.

**Needed for:** Every remaining week in this course. Week 29 immediately applies SFT. Weeks 43–44 apply DPO. Weeks 45–52 apply GRPO.

---

## Common Misconceptions / Pitfalls

- **"Fine-tuning teaches the model new knowledge."** Mostly false. SFT shapes output format and activates existing knowledge for specific input patterns. The model cannot learn facts it never saw in pretraining.
- **"More epochs = better fine-tuning."** False. Overtraining on a small dataset causes catastrophic forgetting and reward hacking. You will learn to watch for this.
- **"Instruction tuning and SFT are different techniques."** No — instruction tuning is SFT on instruction-formatted data. Same loss function.
- **"You need millions of examples to fine-tune."** Not for behavioral shift. InstructGPT used ~13K labeled examples for the SFT stage. Quality >> quantity at this scale.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Watch Karpathy's "Deep Dive into LLMs like ChatGPT" (3h31m) | 3.5h |
| Read InstructGPT paper sections 1–3 | 1h |
| Read HuggingFace LLM Course Chapter 11 intro | 30m |
| Read Karpathy's 2025 Year in Review blog post | 30m |
| Draw the post-training pipeline diagram | 30m |
| Commit diagram to GitHub | 15m |
