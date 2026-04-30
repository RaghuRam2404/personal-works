# Week 60 — GRPO with Executable Rewards (Final)

## Learning Objectives

By the end of this week, you will be able to:

- Configure and run the final GRPO training step starting from your DPO-v3 checkpoint
- Design a multi-signal reward function that combines execution correctness, result accuracy, and SQL quality signals
- Tune GRPO group size, reward scale, and KL coefficient for the SQL domain
- Verify that GRPO improves over the DPO baseline without catastrophic regression
- Produce the capstone model: `postgres-sqlcoder-7b-final`

## GRPO: The Final Alignment Step

You have done GRPO before (Phase 5, Weeks 47–48). This is the final, polished version, starting from a much stronger initialization (DPO-v3 instead of Phase 4 SFT) with a more sophisticated reward function and a refined training procedure.

The key difference from Phase 5: you are now optimizing a model that is already very good at SQL generation. GRPO's role is not to teach SQL from scratch but to push the final performance ceiling by providing exact correctness feedback at inference time. This is the most powerful signal available for text-to-SQL: "run the code and see if it works."

## Concepts

### The Final Reward Function Design

Your reward function must balance multiple objectives. A multi-signal reward works better than binary correctness alone:

```python
def reward(question, generated_sql, reference_sql, schema, conn):
    score = 0.0
    
    # Signal 1: Execution reward (binary — most important)
    exec_result = execute_sql(generated_sql, conn)
    if exec_result.status == "error":
        return -0.5  # syntax/semantic error penalty
    score += 1.0  # executes without error
    
    # Signal 2: Result accuracy (compare result sets)
    ref_result = execute_sql(reference_sql, conn)
    if results_match(exec_result.rows, ref_result.rows):
        score += 1.0  # exact match bonus
    elif results_partially_match(exec_result.rows, ref_result.rows):
        score += 0.3  # partial credit (same schema, different rows)
    
    # Signal 3: Efficiency bonus (optional, use sparingly)
    if is_efficient(generated_sql, conn):  # EXPLAIN cost < threshold
        score += 0.1
    
    # Signal 4: Format reward (ensure proper SQL formatting)
    if starts_with_select_or_with(generated_sql):
        score += 0.1  # not returning natural language instead of SQL
    
    return score  # range: -0.5 to 2.2
```

**Why not use binary reward only?** Binary reward (0 or 1) creates sparse gradients — most early samples get 0 and the model receives no informative signal about what to improve. The multi-signal reward provides denser gradients: a query that executes but returns wrong rows gets 1.0 (not 0), which tells the model "execution structure is good, fix the logic."

### GRPO Group Size and Sampling

GRPO generates K candidates per prompt, computes their rewards, and uses the reward differences within the group as relative advantage estimates (no critic needed). Key parameters:

- **Group size K:** 8 is typical for small models. K=16 gives better advantage estimates but costs 2× the compute.
- **Temperature during sampling:** 0.8–1.0. Must be > 0 (greedy decoding would produce identical candidates, giving no variance).
- **Reward normalization:** Within each group, normalize rewards to zero mean and unit variance before computing loss. This prevents reward scale from dominating.

### KL Coefficient and Stability

GRPO uses a KL penalty to prevent the policy from deviating too far from the reference model:

```
L_GRPO = -E[advantage * log π_θ] + β_kl * KL(π_θ || π_ref)
```

For your final run, start with `kl_coef=0.05`. Too high (> 0.2): model barely moves from DPO checkpoint; GRPO adds nothing. Too low (< 0.01): unstable training; loss diverges.

### The Reference Model for GRPO

Use your DPO-v3 checkpoint as both the starting policy and the reference. This means GRPO will keep the policy close to DPO-v3 while improving it. If you notice GRPO hurting general SQL quality (regression on BIRD-SQL), increase `kl_coef` to pull back toward the DPO reference.

### Final Training Configuration

GRPO on 1,000–2,000 diverse prompts (not your full v3 dataset — GRPO is expensive at K=8 per prompt):

```python
GRPOConfig(
    num_generations=8,          # group size K
    temperature=0.9,            # sampling temperature
    learning_rate=5e-6,         # very low — final alignment step
    kl_coef=0.05,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # 1 prompt × 8 samples = 8 forward passes
    gradient_accumulation_steps=8,  # effective batch = 8 prompts
    max_new_tokens=512,
    report_to="wandb",
    run_name="week-60-grpo-final",
)
```

Total compute: 2,000 prompts × 8 samples = 16,000 inference + backward passes. At H100: ~3–4 hours. Budget: ~$9–12.

### Evaluating GRPO Progress During Training

Log every 50 steps:
- Mean reward per group (should increase from ~0.8 to ~1.5 over training)
- Execution rate on generated samples (should increase toward 85–90%)
- KL divergence from reference (should stay bounded, < 5 bits typically)
- Domain execution accuracy on your held-out eval (the truth metric)

Stop training early if KL divergence exceeds 10 bits — the policy has drifted too far.

### The Capstone Model

After GRPO, merge LoRA adapters and push the final model:

```python
model.save_pretrained_merged(
    "<your-handle>/postgres-sqlcoder-7b-final",
    tokenizer,
    save_method="merged_16bit",  # full bf16 weights
)
```

This is your capstone: `postgres-sqlcoder-7b-final`. This is the model that goes into the quantization pipeline (Weeks 63–64), the evaluation harness (Weeks 61–62), and the technical report.

## Connections

This week is the final application of GRPO, which you first encountered in Weeks 47–48 (the Phase 5 GRPO sprint on REINFORCE-style alignment). The key difference is initialization: here you start from DPO-v3 (Week 59) rather than a Phase 4 SFT checkpoint, giving GRPO a much stronger starting point and letting it focus on the final performance ceiling rather than teaching SQL from scratch.

Week 59 (DPO v3) produced the checkpoint you are fine-tuning this week. Your DPO-v3 model should already score 65–70% execution accuracy on your eval set; GRPO's job is to push that toward 75–80% using live execution feedback.

Weeks 61–62 will evaluate `postgres-sqlcoder-7b-final` (the model you produce this week) head-to-head against GPT-4o, Claude 3.5 Sonnet, and SQLCoder-7B on BIRD-SQL and Spider 2.0. If GRPO introduced regressions on general SQL, they will show up there — that is why KL regularization is critical.

Weeks 63–64 will quantize this model (GGUF and AWQ) for CPU/edge deployment. The merged 16-bit weights you produce at the end of this week are the input to the quantization pipeline.

### Common Misconceptions and Pitfalls

**"GRPO always improves over DPO."** Not guaranteed. If your DPO-v3 model is already near the ceiling for your data distribution, GRPO may yield only 1–2pp. The returns diminish as the model approaches the performance ceiling imposed by training data quality.

**"I should use all 25K v3 examples as GRPO prompts."** GRPO is 8× more expensive than SFT per prompt (you generate K=8 completions). Running GRPO on 25K prompts would take days on H100. Select 1,000–2,000 diverse, hard prompts from your eval distribution.

**"The reward function needs to be complex."** Keep it simple. The execution correctness signal is the most reliable. Add partial credit for partial correctness. Avoid rewarding length or superficial style — these create reward hacking.

## Time Allocation (6–8 hrs)

- 1h: Finalize reward function; run on 100 examples to verify it works correctly
- 0.5h: Select 1,500 diverse GRPO training prompts from your eval distribution
- 0.5h: Configure GRPO script; smoke test on 50 prompts locally
- 0.5h: Spin up RunPod H100; upload
- 3.5h: Run GRPO on RunPod (~3.5 hours); monitor W&B
- 0.5h: Evaluate final model; compare to DPO baseline
- 0.5h: Merge adapters; push `postgres-sqlcoder-7b-final` to HuggingFace; terminate RunPod
