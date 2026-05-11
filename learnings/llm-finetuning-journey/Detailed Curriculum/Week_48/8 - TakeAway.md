# Week 48 TakeAway — Running GRPO

**One-liner:** Monitor mean_reward and reward_std together; flat reward_std means zero gradient, not convergence.

---

## GRPO Launch Sequence

```bash
# On RunPod A100
pip install unsloth trl transformers datasets peft wandb psycopg2-binary
huggingface-cli login
export WANDB_API_KEY=...
screen -S grpo   # Run inside screen so disconnect doesn't kill training
python grpo_train.py
# Detach: Ctrl+A then D
# Reattach: screen -r grpo
```

---

## Key Monitoring Metrics

| Metric | Healthy | Warning | Action |
|---|---|---|---|
| `mean_reward` | Increasing | Flat > 100 steps | Check reward_std |
| `reward_std` | > 0.1 | < 0.02 | Harder prompts or higher temperature |
| `kl_divergence` | < 5 nats | > 10 nats | Increase β |
| `grad_norm` | < 1.0 | > 5.0 | Stop, reduce LR, restart from checkpoint |
| `loss` | Fluctuating | NaN | Stop immediately, restart with max_grad_norm=0.5 |

---

## Three-Way Eval Template

```python
models = {"v1": v1_path, "v2": v2_path, "v3": v3_path}
results = {}
for name, path in models.items():
    model = load_model(path)
    preds = [model.generate(p) for p in test_prompts]
    sqls = [extract_sql(p) for p in preds]
    exec_results = [execute_sql(s, db_dsn) for s in sqls]
    results[name] = {
        "exec_acc": mean(r["success"] for r in exec_results),
        "sem_acc": mean(r["rows"] == ref for r, ref in zip(exec_results, references)),
        "syntax_err_rate": mean("syntax" in r.get("error","") for r in exec_results),
    }
```

---

## Memory Configuration

| GPU | Recommended config |
|---|---|
| A100 80GB | bf16 for training model, 4-bit for ref model, K=8 |
| A100 40GB | 4-bit for both models, K=8 (may need K=4) |
| A10G 24GB | 4-bit for both, K=4 only |

---

## Numbers to Remember

- Target mean_reward at step 1000: 0.40–0.55 (up from ~0.20 baseline)
- Target execution acc improvement over v2: ≥ 5pp
- Training time (A100 80GB, K=8, 1000 steps): ~6–10 hours
- Cost on RunPod A100 80GB: ~$2.50–4/hour → $15–40 for full run
- Checkpoint every 50 steps; keep last 5 only
- KL healthy ceiling: 5 nats; warning zone: 5–10; stop zone: > 15

---

## Red Flags

- Step time > 200 seconds: K too high or generation too long — reduce K or max_completion_length
- reward_std = 0 at step 50+: model collapsed to single output — increase temperature
- mean_reward below diagnostic baseline at step 100: reward function has a bug
- NaN loss: stop immediately; restart from last checkpoint with `max_grad_norm=0.5` and 10× lower LR
