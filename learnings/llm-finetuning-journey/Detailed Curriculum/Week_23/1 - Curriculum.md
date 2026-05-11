# Week 23 — Evaluation 101: Perplexity, lm-evaluation-harness, MMLU, HellaSwag, ARC

## Learning Objectives

By the end of this week, you will be able to:

- Explain the difference between perplexity-based evaluation and task-based downstream evaluation
- Install and run `lm-evaluation-harness` on your 50M model and on `gpt2`
- Interpret results from MMLU, HellaSwag, and ARC benchmarks
- Explain why a model can have low perplexity but still fail downstream tasks
- Design a domain-specific evaluation appropriate for your PostgreSQL/TimescaleDB target

---

## Concepts

### Two Kinds of Language Model Evaluation

**Perplexity-based evaluation** (Weeks 21–22):
- Measures how well the model predicts held-out text
- Language-model native: works on any text, no task structure needed
- Lower is better
- Problem: does not tell you if the model can do useful tasks

**Downstream task evaluation:**
- Measures accuracy on a specific task (multiple choice, text classification, code generation, etc.)
- Directly measures capability that users care about
- Problem: requires a labeled dataset for each task; cannot cover everything

Both are necessary. A model with good perplexity but poor downstream performance has learned the surface statistics of text without understanding. A model with good downstream performance but high perplexity is unusual but possible (highly specialized models).

### EleutherAI lm-evaluation-harness

The [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is the standard open-source evaluation framework for LLMs. It implements:
- 60+ benchmarks including MMLU, HellaSwag, ARC, WinoGrande, TruthfulQA, GSM8K
- Both log-likelihood scoring (for multiple-choice) and generation-based evaluation
- HuggingFace model integration: pass a model name or local path

**Installation:**
```bash
pip install lm-eval
```

**Basic usage:**
```bash
# Evaluate GPT-2 on HellaSwag and ARC-Easy
lm_eval \
  --model hf \
  --model_args pretrained=gpt2 \
  --tasks hellaswag,arc_easy \
  --device cuda:0 \
  --batch_size 16 \
  --output_path results/gpt2_eval.json

# Evaluate your own model (from local path or HuggingFace Hub)
lm_eval \
  --model hf \
  --model_args pretrained=<your-handle>/fineweb-50m-pretrain \
  --tasks hellaswag,arc_easy,mmlu \
  --device cuda:0 \
  --batch_size 4 \
  --output_path results/your_model_eval.json
```

### Understanding the Key Benchmarks

**HellaSwag:**
- 70,000 multiple-choice questions about "what happens next" in a video clip description
- 4 options per question; models choose the most plausible continuation
- Evaluated by log-likelihood: model scores each option and picks the highest
- Random baseline: 25%. Human: ~95%. GPT-2 (124M): ~31%. GPT-4: ~95%
- Measures: world knowledge and common-sense reasoning via text

**ARC (AI2 Reasoning Challenge):**
- Two sets: ARC-Easy and ARC-Challenge (ARC-E and ARC-C)
- Elementary school science questions in multiple-choice format
- ARC-Easy: straightforward questions answerable from simple facts
- ARC-Challenge: questions requiring multi-hop reasoning
- Random baseline: 25% (4-choice). Human: ~80%. GPT-2 (124M): ~44% (ARC-E). GPT-4: ~96%

**MMLU (Massive Multitask Language Understanding):**
- 57 subjects across STEM, humanities, social science, law, medicine
- 14,000+ questions, 4-choice multiple choice
- Random baseline: 25%. Human expert: ~89%. GPT-2 (124M): ~26% (near random). GPT-4: ~87%
- For a 50M model: expect 23–28% (near random) — this is expected and fine

**Why your 50M model will score near random on MMLU:**
MMLU requires factual knowledge, reasoning, and domain expertise. Your 50M model has ~56M parameters to store ~700GB of training data's "knowledge" — information capacity is the bottleneck. GPT-3 (175B params) gets 43% on MMLU. Getting above 30% at 50M params requires much more training data and careful data selection.

### How Log-Likelihood Evaluation Works

For multiple-choice benchmarks, lm-evaluation-harness does not use the model's generation. Instead:

```
For question Q with options A, B, C, D:
1. Concatenate: "Q + A", "Q + B", "Q + C", "Q + D"
2. Compute log P(option | Q) for each
3. Select the option with the highest log-probability
```

This avoids the need for the model to "know to output A, B, C, or D" — any causal language model can be evaluated this way regardless of instruction-following ability.

**Normalization:** Some benchmarks normalize by length (log P / length) to prevent bias toward short answers. HellaSwag uses this; MMLU typically does not.

### Perplexity vs. Downstream Task Performance

Consider two models:
- Model A: trained on Wikipedia, excellent at factual recall, PPL=18
- Model B: trained on diverse web text, poorer factual recall, PPL=21

Model A might outperform Model B on MMLU (factual questions) but underperform on HellaSwag (narrative common sense). Perplexity on a specific distribution predicts task accuracy on related tasks, not all tasks.

This is why modern LLM evaluation uses many diverse tasks: no single metric captures overall capability.

### Designing a Domain-Specific Evaluation (Preview for Phase 6)

For your PostgreSQL/TimescaleDB goal, standard benchmarks are insufficient — they do not measure SQL accuracy. You need:

1. **Text-to-SQL accuracy**: Execute the generated SQL on a test database and check if the result matches the reference answer
2. **Execution rate**: What fraction of generated SQL runs without syntax errors?
3. **Semantic similarity**: For SQL that runs, does it return the correct rows?
4. **Spider/BIRD accuracy**: Established NL→SQL benchmarks; run as a proxy for your domain

You will build this domain evaluation in Phases 5–6. For now, understand that standard benchmarks like MMLU measure nothing useful for your goal.

---

## Connections

**Week 22:** You computed perplexity manually. This week you automate it and add downstream task accuracy.

**Week 24:** Understanding the benchmarks that SOTA models are evaluated on is essential for reading the Llama 3, Qwen2.5, and DeepSeek papers in Week 24.

**Phase 6:** Your final evaluation will use a custom benchmark you build, plus Spider and BIRD as proxies.

---

## Common Misconceptions

- **"MMLU accuracy measures overall intelligence."** MMLU measures factual recall and reasoning in 57 specific domains. A model could score perfectly on MMLU and still fail at code generation, creative writing, or instruction following.
- **"Higher benchmark scores always mean better models."** Benchmark contamination (test data in training set) inflates scores. Check whether your training data includes the test sets.
- **"My 50M model scoring 25% on MMLU is a failure."** It is not — 25% is near random, which is expected for a small model without enough parameter capacity for factual knowledge.
- **"lm-evaluation-harness evaluates my model's generation quality."** For most tasks in the harness, it evaluates log-likelihood scoring, not generation. A model that generates incoherent text can still score well on multiple-choice if its likelihood function is well-calibrated.

---

## Time Allocation (6–8 hrs)

- 0.5h: Install lm-eval, verify it works on a quick GPT-2 run (5 min sanity check)
- 1.5h: Run full evaluation on GPT-2 (HellaSwag + ARC-E + ARC-C + MMLU-5shot subset)
- 1.5h: Run full evaluation on your 50M model
- 1h: Interpret and compare results; write evaluation table
- 1.5h: Write the comparison report (`week-23-eval-report.md`)
- 0.5h: Commit and journal entry
