# Week 60 Assignment — Final GRPO Training

## Setup Checklist

- [ ] DPO-v3 checkpoint: `<your-handle>/qwen2.5-coder-7b-postgres-dpo-v3`
- [ ] PostgreSQL with all test schemas running
- [ ] RunPod H100 access; budget ~$12 for this run
- [ ] `trl>=0.8` with GRPOTrainer; Unsloth
- [ ] 200-example custom eval set (plus 100-example held-out for honest eval)
- [ ] W&B project `week-60-grpo` created

---

## Task 1 — Finalize and Test the Reward Function

**Goal:** Verify your reward function produces correct, informative rewards before starting training.

**Requirements:**
Write `reward_fn.py` with:

```python
def compute_reward(prompt, completion, reference_sql, schema_ddl, conn) -> float:
    sql = extract_sql(completion)  # strip markdown fences if present
    if sql is None:
        return -1.0  # didn't generate SQL at all
    
    # Execute
    try:
        with conn.cursor() as cur:
            cur.execute("BEGIN")
            cur.execute(schema_ddl)  # ensure schema exists in transaction
            cur.execute(sql)
            rows = cur.fetchall()
            conn.rollback()
        exec_reward = 1.0
    except Exception:
        conn.rollback()
        return -0.5
    
    # Result accuracy
    ref_rows = execute_reference(reference_sql, conn)
    if rows == ref_rows or set(map(str, rows)) == set(map(str, ref_rows)):
        accuracy_reward = 1.0
    elif len(rows) == len(ref_rows):  # right shape, wrong values
        accuracy_reward = 0.2
    else:
        accuracy_reward = 0.0
    
    # Format reward
    format_reward = 0.1 if sql.strip().upper().startswith(("SELECT","WITH","INSERT","UPDATE","DELETE")) else 0.0
    
    return exec_reward + accuracy_reward + format_reward
```

Test on 100 examples from your eval set:
- Run reference SQL → compute its own reward (should be 2.1)
- Run completely wrong SQL → should be -0.5
- Run correct-execution/wrong-result SQL → should be 1.0–1.2
- Log the reward distribution histogram to W&B

**Deliverable:** `reward_fn.py` with 100-example test committed.

---

## Task 2 — Select GRPO Training Prompts

**Goal:** Choose 1,500 diverse, challenging prompts for GRPO optimization.

**Requirements:**
Write `select_grpo_prompts.py` that:
- Loads your v3 dataset + your custom eval set
- Runs your DPO-v3 model with greedy decoding on each example
- Selects examples where the model's greedy output fails (execution status = fail OR wrong rows): these are the hardest cases where GRPO has the most room to improve
- Ensures skill diversity: include at least 300 TimescaleDB examples, 200 multi-turn examples
- Saves as `grpo_prompts.jsonl` with fields: `prompt`, `reference_sql`, `schema_ddl`

**Reasoning:** GRPO is most effective on prompts where the current model fails. Training on examples the model already gets right produces near-zero reward variance — no useful gradient.

**Deliverable:** `grpo_prompts.jsonl` with ≥ 1,500 examples committed.

---

## Task 3 — GRPO Training Script

**Goal:** Run the final GRPO alignment step.

**Requirements:**
Write `train_grpo_final.py`:

```python
from trl import GRPOTrainer, GRPOConfig

def make_reward_fn(conn):
    def reward_fn(completions, prompts, **kwargs):
        rewards = []
        for completion, prompt_data in zip(completions, kwargs["reference_data"]):
            r = compute_reward(
                prompt=prompt_data["prompt"],
                completion=completion,
                reference_sql=prompt_data["reference_sql"],
                schema_ddl=prompt_data["schema_ddl"],
                conn=conn
            )
            rewards.append(r)
        return rewards
    return reward_fn

grpo_config = GRPOConfig(
    num_generations=8,
    temperature=0.9,
    learning_rate=5e-6,
    kl_coef=0.05,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_new_tokens=512,
    output_dir="./grpo_final_output",
    run_name="week-60-grpo-final",
    report_to="wandb",
    logging_steps=10,
    save_steps=100,
)
```

Log per-step:
- `mean_reward` (should increase over training)
- `execution_rate` (fraction of K samples that execute per group)
- `exact_match_rate` (fraction that match reference result)
- `kl_divergence` (should stay < 10 bits)
- Every 100 steps: domain execution accuracy on held-out 100-example eval

**Terminate RunPod immediately after training.**

**Deliverable:** GRPO training completes; model merged and pushed as `<your-handle>/postgres-sqlcoder-7b-final`.

---

## Task 4 — Final Capstone Evaluation

**Goal:** Quantify the final model's performance across the full training pipeline.

**Requirements:**
Create `final_model_comparison.md` showing:

| Model | Custom 200 | TimescaleDB subset | BIRD-SQL dev sample |
|-------|-----------|-------------------|---------------------|
| Base Qwen2.5-Coder-7B | X% | X% | X% |
| Phase 5 GRPO (v3 baseline) | X% | X% | X% |
| Week 58 SFT-v3 | X% | X% | X% |
| Week 59 DPO-v3 | X% | X% | X% |
| Week 60 GRPO-final | X% | X% | X% |

Use identical prompts, identical eval code, identical Postgres connection for all models.

**Acceptance criteria:**
- GRPO-final ≥ DPO-v3 on custom benchmark (no regression)
- GRPO-final ≥ Phase 5 GRPO + 10pp on custom benchmark
- TimescaleDB subset accuracy ≥ 60%

**Deliverable:** `final_model_comparison.md` committed. This table goes directly into your technical report (Week 69).

---

## Stretch Goals

- Run GRPO with two different kl_coef values (0.05 vs 0.1) for 200 steps each; compare mean reward and KL divergence
- Profile inference speed of the final merged model: tokens/second on H100, A100, and your Mac (MPS)
- Compute the "correctness at K" (pass@K) metric: for each eval prompt, sample K=8 candidates; what fraction of prompts have at least 1 correct answer? This is the upper bound on what GRPO can achieve.
