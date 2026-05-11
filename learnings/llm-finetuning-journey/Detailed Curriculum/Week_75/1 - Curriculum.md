# Week 75 — Iteration Polish: Trying Different Base Models

## Learning Objectives

By the end of this week, you will be able to:

- Design a controlled experiment that isolates base model quality from fine-tuning quality
- Run your full SFT pipeline on a new base model with minimal code changes
- Compare two fine-tuned models on the same benchmark with the same inference settings
- Explain why the best zero-shot base model is not always the best fine-tuned model
- Produce a model comparison table suitable for inclusion in your technical report

## Concepts

### Why Base Model Choice Matters More Than You Think

Your postgres-sqlcoder-7b was built on Qwen2.5-Coder-7B-Instruct. That choice was motivated by: strong SQL benchmarks, permissive Apache 2.0 license, good Unsloth support, and a code-optimized pretraining corpus. But the field moves fast. In the months since you started this course, Llama 3.1 8B, Gemma 2 9B, and DeepSeek-Coder-V2-Lite have all been released or updated. This week you empirically answer: was your base model choice optimal?

The critical insight: zero-shot base model performance on SQL benchmarks is a weak predictor of fine-tuned model performance. The relevant quantity is not "how good is the base model at SQL?" but "how much can the base model learn from your 25K training examples?" A model with strong meta-learning ability and good representations may fine-tune to higher accuracy even if its zero-shot SQL is lower.

### Designing the Controlled Experiment

A valid base model comparison requires holding everything else constant:

- Same training data (your v3 SFT dataset, 25.5K examples)
- Same hyperparameters (LR=2e-4, LoRA r=64, alpha=128, same schedule)
- Same number of training steps (2,400)
- Same evaluation benchmark and metric
- Same inference settings (temperature=0.1, max_tokens=512, same prompt template)

The only variable is the base model. If you change anything else between runs, your comparison is confounded.

One practical challenge: the prompt template must match each model's expected chat format. Llama 3.1 uses a different chat template than Qwen2.5 or Gemma. Your training data preparation script must apply the correct template for each model:

```python
from transformers import AutoTokenizer

def format_for_model(example, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example)},
        {"role": "assistant", "content": example["sql"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)
```

This is the most failure-prone step — a model fine-tuned with the wrong chat template will appear to underperform not because of model quality but because of format mismatch.

### Candidate Base Models

Three candidates to evaluate (plus your existing Qwen2.5-Coder-7B baseline):

**Llama 3.1 8B Instruct**
- Architecture: dense transformer, 8B parameters, GQA
- Context: 128K tokens (RoPE with rope_scaling)
- License: Llama 3.1 Community License (allows commercial use for most entities)
- Zero-shot SQL: approximately 71% on Spider 1.0 (published)
- Unsloth support: full
- Key risk: Llama 3.1 was not specifically trained on SQL; code knowledge comes from general pretraining only

**Gemma 2 9B Instruct**
- Architecture: dense transformer, 9B parameters, sliding window + global attention
- Context: 8K tokens
- License: Gemma Terms of Use (permissive but non-Apache)
- Zero-shot SQL: approximately 69% on Spider 1.0
- Unsloth support: full
- Key risk: 8K context may limit large schema handling; sliding window attention may require adaptation for very long SQL prompts

**DeepSeek-R1-Distill-Qwen-7B**
- Architecture: dense Qwen2.5 architecture (same as your existing model), 7B parameters
- Context: 32K tokens
- License: MIT
- Zero-shot SQL: strong due to Qwen2.5 base and R1 reasoning distillation
- Unsloth support: full (same architecture as Qwen)
- Key advantage: chain-of-thought reasoning from R1 distillation may help complex SQL; same Qwen architecture means your existing scripts work unchanged

### Running the Comparison

The practical workflow for each candidate:

```bash
# 1. Prepare training data with correct chat template
python prepare_data.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output data/sft_llama31.jsonl

# 2. Run SFT (same hyperparameters, different --model_name)
python train_sft.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data data/sft_llama31.jsonl \
    --output checkpoints/llama31-sqlcoder \
    --lr 2e-4 --lora_r 64 --steps 2400

# 3. Evaluate on Custom-200 benchmark
python eval.py \
    --model checkpoints/llama31-sqlcoder \
    --benchmark data/custom_200.json \
    --output results/llama31_results.json
```

Repeat for each candidate. Total compute: approximately 2–3 GPU-hours per model × 3 candidates = 6–9 GPU-hours.

### Interpreting Results

The comparison table you produce should look like:

| Base Model | Params | Zero-shot SQL | After SFT | Delta | BIRD-SQL | VRAM (Q4) |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-7B (existing) | 7B | 61.0% | 83.1% | +22.1 | 68.4% | 4.5 GB |
| Llama 3.1 8B | 8B | 57.4% | ?% | ? | ? | 4.8 GB |
| Gemma 2 9B | 9B | 55.1% | ?% | ? | ? | 5.3 GB |
| R1-Distill-Qwen-7B | 7B | 63.2% | ?% | ? | ? | 4.5 GB |

The "Delta" column is most informative: how much did SFT lift each model? A base model with a small delta is near-saturated on your training data; a model with a large delta is efficiently learning from your domain examples.

If one candidate achieves 84%+ on Custom-200, it is your new best model and should be quantized and pushed to Hub as `postgres-sqlcoder-v2`.

### When to Stop and Declare the Winner

You have one week. Prioritize:
1. R1-Distill-Qwen-7B first (highest expected performance, lowest toolchain risk)
2. Llama 3.1 8B second (strongest brand, widest community support)
3. Gemma 2 9B third (lower priority due to 8K context limitation)

If compute runs out: compare on a 50-example subset of Custom-200 for fast iteration, then run the full 200-example eval only on the winner.

## Connections

This week's comparison table feeds directly into your technical report as an updated Table 1 variant (if you add a "base model ablation" row). The winner of this comparison becomes the base for Weeks 76 and 77 if it significantly outperforms your existing Qwen2.5 model. Otherwise, stick with your existing model to minimize compounding changes.

## Common Misconceptions / Pitfalls

The most common mistake is comparing fine-tuned models without verifying that the chat template was applied correctly during training. A wrong template is silent — the model trains without errors but the format mismatch causes systematic accuracy degradation. Always verify: print 3 training examples after template application and confirm they match the model's expected format.

Do not run DPO and GRPO for every candidate in Week 75 — SFT comparison is sufficient to rank base models. Full pipeline re-training is expensive; only apply to the winner.

## Time Allocation (6–8 hours)

- 0.5h: Verify chat templates for each candidate model
- 1.0h: Prepare training data for each model (3 data prep runs)
- 3.0h: Run SFT for each candidate (sequential on one GPU; parallel if you have multiple)
- 1.5h: Evaluate all four models (existing + 3 candidates) on Custom-200
- 1.0h: Write comparison table, analysis, and decision in `results/base_model_comparison.md`
