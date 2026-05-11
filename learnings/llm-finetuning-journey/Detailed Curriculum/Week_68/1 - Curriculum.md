# Week 68 — Technical Report Week 2: Training Pipeline and Architecture

## Learning Objectives

By the end of this week, you will be able to:

- Write a training pipeline section that documents all four training stages with reproducibility-grade detail
- Describe architecture decisions clearly without redundant base-model description
- Present hyperparameter tables in the format expected by ML publication venues
- Write a compute budget section that translates your training runs into GPU-hours
- Integrate the training section into your growing report document

## Concepts

### What the Training Pipeline Section Must Cover

The training section answers: what did you do to the base model, in what order, with what hyperparameters, and how long did it take? Readers need enough detail to reproduce your pipeline. The section should not re-explain how transformers work — assume the reader knows. Instead, describe your specific choices and why you made them.

Structure:

```
4. Training Pipeline

4.1 Base Model
4.2 Stage 1: Continued Pretraining (CPT)
4.3 Stage 2: Supervised Fine-Tuning (SFT)
4.4 Stage 3: Direct Preference Optimization (DPO)
4.5 Stage 4: Group Relative Policy Optimization (GRPO)
4.6 Compute Budget
```

### Writing the Base Model Subsection

This should be one paragraph, not a tutorial. State: model name, parameter count, key architectural features relevant to your task, and why you chose it over alternatives.

"We fine-tune Qwen2.5-Coder-7B-Instruct (Hui et al. 2024), a 7.6B parameter decoder-only transformer with grouped query attention (GQA) and a 128K context window. We select this model for three reasons: strong baseline SQL performance on Spider 1.0 (76.8%), native support for long schema prompts due to the 128K context, and permissive licensing (Apache 2.0) enabling model release."

Do not describe the transformer architecture in detail — cite the original paper. Do describe the specific features that motivated your choice.

### Writing Each Training Stage

For each of the four stages, you need: (a) objective/loss function, (b) dataset, (c) hyperparameters, (d) training duration, (e) what you observed or measured.

Stage 1 (CPT) template:
```
4.2 Continued Pretraining

We continue pretraining from Qwen2.5-Coder-7B on 102M tokens of
PostgreSQL-domain text using a standard causal language modeling objective:

  L_CPT = -sum_t log p(x_t | x_{<t})

We train for 1 epoch with a cosine learning rate schedule, warmup over
the first 2% of steps, and a peak LR of 2e-5. LoRA adapters (r=64,
alpha=128) are applied to all linear layers. Batch size 32, sequence
length 2048. Training runs for approximately 1,600 steps on 2× A100-40GB
GPUs, requiring 4.2 GPU-hours. Validation perplexity on a held-out SQL
corpus decreases from 18.4 (base model) to 11.2 (post-CPT).
```

For stages 2–4, follow the same pattern. The key numbers for each stage:
- SFT: learning rate, LoRA rank, batch size, steps, final val loss
- DPO: beta, learning rate, steps, reward margin achieved
- GRPO: K (samples per prompt), reward function formula, steps, accuracy gain

### Hyperparameter Tables

Every paper that describes training must include a hyperparameter table. Format:

| Hyperparameter | CPT | SFT | DPO | GRPO |
|---|---|---|---|---|
| Learning rate | 2e-5 | 2e-4 | 5e-6 | 1e-6 |
| LR schedule | cosine | cosine | linear | constant |
| Warmup steps | 32 | 100 | 50 | 0 |
| Batch size | 32 | 16 | 8 | 4 |
| LoRA rank | 64 | 64 | 64 | 64 |
| LoRA alpha | 128 | 128 | 128 | 128 |
| LoRA target modules | all-linear | all-linear | all-linear | all-linear |
| Max sequence length | 2048 | 2048 | 1024+1024 | 1024 |
| Training steps | 1,600 | 2,400 | 800 | 600 |
| GPU type | A100-40GB | A100-40GB | A100-40GB | A100-40GB |
| GPU count | 2 | 2 | 1 | 1 |

This table lives in the main text AND is duplicated in the Appendix with more detail.

### Writing the Compute Budget

The compute budget section answers two questions: how much compute did you use (for reproducibility), and could a researcher with a smaller budget reproduce your results?

Report GPU-hours per stage, total cost (use RunPod or Lambda Labs pricing), and minimum required setup. Example:

"Total training compute: 12.4 GPU-hours on A100-40GB (approximate RunPod cost at $1.40/hr: $17.36). CPT required 2× A100 for 4.2h (8.4 GPU-h); SFT required 2× A100 for 2.1h (4.2h GPU-h); DPO required 1× A100 for 0.8h; GRPO required 1× A100 for 1.5h. A researcher with access to a single A100-40GB or equivalent (e.g., Google Colab Pro+) can reproduce all stages sequentially at approximately double the wall-clock time."

### Describing Architecture Decisions

Architecture decisions are choices you made beyond the base model: LoRA rank selection, target modules, sequence length, chat template. For each decision, briefly justify it:

"We apply LoRA to all linear projection layers including attention (Q, K, V, O) and FFN (gate, up, down) with rank r=64. Preliminary experiments showed rank 32 gave 1.8 pp lower accuracy on the SFT validation set; rank 128 showed no improvement over 64 (Table A1 in Appendix). We use alpha=128 (2× rank) following the convention established in QLoRA (Dettmers et al. 2023)."

## Connections

The training section directly enables Weeks 69–70 (evaluation and ablations). Your ablation study (Week 69) will reference the training stages described here — "Stage 2-only" vs "Stage 2+3" vs "full pipeline." The hyperparameter table in the Appendix is the artifact that reproducibility reviewers check first.

## Common Misconceptions / Pitfalls

Do not describe the theory of DPO or GRPO in detail — cite the original papers and describe only your implementation choices. Readers who need the theory will read the original paper. Your space is for what you specifically did.

Do not omit GPU type and count from the compute section. "Took 4 hours" with no hardware specification is useless for reproducibility.

Do not give "ballpark" hyperparameters. Report exact values — these come directly from your W&B config logs.

## Time Allocation (6–8 hours)

- 0.5h: Review W&B config logs from Weeks 57–60 and extract exact hyperparameters
- 1.0h: Write base model + CPT subsection
- 1.5h: Write SFT + DPO subsections
- 1.0h: Write GRPO subsection
- 1.0h: Build hyperparameter table
- 0.5h: Write compute budget section
- 0.5h: Integrate into main `report.md` document and review for consistency
